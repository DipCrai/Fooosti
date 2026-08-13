import asyncio
import base64
import hmac
import json
import os
import subprocess
import threading
import time
import traceback
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from modules import constants, worker_proc

OUT_DIR = os.environ.get('FOOOSTI_TMP_DIR', constants.FOOOSTI_TMP_DIR)
KEEPALIVE_MINUTES = int(os.environ.get('FOOOSTI_KEEPALIVE_MINUTES', '0') or '0')
API_TOKEN = os.environ.get('FOOOSTI_API_TOKEN', '')
GENERATION_TIMEOUT = int(os.environ.get('FOOOSTI_GENERATION_TIMEOUT') or constants.FOOOSTI_GENERATION_TIMEOUT)

# worker scratch dir for ndarray outputs, kept under our control dir so it is
# always tied to this API instance (never a guessed default like /tmp/fooocus)
TEMP_DIR = os.path.join(OUT_DIR, 'temp')

os.makedirs(OUT_DIR, exist_ok=True)

if not API_TOKEN:
    print('[Fooosti] WARNING: FOOOSTI_API_TOKEN is not set - /sdapi/v1/* endpoints are unauthenticated.', flush=True)

_worker = None
_worker_lock = threading.RLock()
_busy = False
_last_activity = 0.0


class _BusyError(Exception):
    pass


class Txt2ImgRequest(BaseModel):
    prompt: str = Field(default="", max_length=10000)
    negative_prompt: str = Field(default="", max_length=10000)
    steps: int | None = Field(default=None, ge=1, le=200)
    width: int | None = Field(default=None, ge=16, le=2048)
    height: int | None = Field(default=None, ge=16, le=2048)
    batch_size: int | None = Field(default=None, ge=1, le=8)
    cfg_scale: float | None = Field(default=None, ge=1.0, le=30.0)
    seed: int = -1
    sampler_name: str | None = None
    scheduler_name: str | None = None
    style_selections: list[str] | None = None
    performance: str | None = None
    base_model_name: str | None = None
    sharpness: float | None = None
    metadata_scheme: str | None = None


def _spawn_worker():
    global _worker
    # the API worker must not write into the WebUI's history log or outputs,
    # and must use our own scratch dir so cleanup is predictable
    _worker = worker_proc.spawn('--disable-image-log', '--disable-metadata',
                                '--temp-path', TEMP_DIR,
                                stdout=subprocess.DEVNULL)
    print(f'[Fooosti] worker spawned pid={_worker.pid}', flush=True)
    return _worker


def _get_worker():
    if not worker_proc.alive(_worker):
        _spawn_worker()
    return _worker


def _kill_worker():
    global _worker
    if worker_proc.terminate(_worker):
        print(f'[Fooosti] worker pid={_worker.pid} terminated', flush=True)
    _worker = None


def _read_resp_file(resp_file: str) -> dict:
    # the worker writes resp_*.json with an atomic replace now, but keep a
    # short retry so a half-visible file can never 500 the request
    for _ in range(10):
        try:
            with open(resp_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            time.sleep(0.2)
    with open(resp_file) as f:
        return json.load(f)


def _cleanup_failed(resp_file: str):
    try:
        if os.path.exists(resp_file):
            os.remove(resp_file)
    except Exception:
        pass
    # orphaned ndarray outputs the worker wrote into TEMP_DIR on a failed job
    import glob
    try:
        for p in glob.glob(os.path.join(TEMP_DIR, 'api_*.png')):
            try:
                os.remove(p)
            except Exception:
                pass
    except Exception:
        pass


def _submit_task(req: Txt2ImgRequest) -> dict:
    global _last_activity, _busy
    resp_file = os.path.join(OUT_DIR, f'resp_{uuid.uuid4().hex}.json')

    msg = {'task': req.model_dump(), 'resp_file': resp_file}

    # run in an executor thread; the lock is held only for the check+set and
    # never across the generation, so the event loop stays responsive and the
    # 429 gate is reliable (single owner of _busy)
    with _worker_lock:
        if _busy:
            raise _BusyError('generation already in progress')
        _busy = True
        _last_activity = time.time()

    try:
        worker = None
        for attempt in range(2):
            worker = _get_worker()
            try:
                worker.stdin.write((json.dumps(msg) + '\n').encode())
                worker.stdin.flush()
                break
            except Exception:
                # worker may have died / stdin closed between checks
                _kill_worker()
                if attempt == 0:
                    continue
                raise RuntimeError('generation worker failed to start')

        deadline = time.time() + GENERATION_TIMEOUT
        while time.time() < deadline:
            if os.path.exists(resp_file):
                resp = _read_resp_file(resp_file)
                try:
                    os.remove(resp_file)
                except Exception:
                    pass
                return resp
            if worker.poll() is not None:
                raise RuntimeError(f'generation worker died (pid={worker.pid})')
            time.sleep(0.3)

        raise TimeoutError('Fooosti generation timed out')
    except Exception:
        _cleanup_failed(resp_file)
        raise
    finally:
        with _worker_lock:
            _busy = False
            _last_activity = time.time()
            if KEEPALIVE_MINUTES <= 0:
                _kill_worker()


def _user_config():
    cfg_path = os.environ.get('config_path', os.path.join(os.environ.get('DATADIR', '/content/data'), 'config.txt'))
    try:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
            return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def _checkpoints_dir():
    cp = _user_config().get('path_checkpoints')
    if isinstance(cp, list):
        return cp[0] if cp else None
    if isinstance(cp, str):
        return cp
    return os.environ.get('path_checkpoints', '/content/data/models/checkpoints/')


def _checkpoint_filenames():
    d = _checkpoints_dir()
    try:
        return sorted(f for f in os.listdir(d)
                      if f.endswith('.safetensors') or f.endswith('.ckpt'))
    except Exception:
        return []


def _default_checkpoint():
    dm = _user_config().get('default_model')
    files = _checkpoint_filenames()
    if dm and dm in files:
        return dm
    return files[0] if files else (dm or 'None')


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    _kill_worker()


app = FastAPI(title='Fooosti', version='0.1.0', lifespan=lifespan)


threading.Thread(
    target=worker_proc.keepalive_reaper,
    args=('Fooosti',
          lambda: _worker,
          lambda: _busy,
          lambda: time.time() - _last_activity,
          _kill_worker),
    daemon=True).start()


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.api_route('/', methods=['GET', 'HEAD'])
def index():
    return PlainTextResponse('Fooosti is running')


def _require_api_token(x_api_token: str = Header(default='')):
    if API_TOKEN and not hmac.compare_digest(x_api_token, API_TOKEN):
        raise HTTPException(status_code=401, detail='invalid API token')


@app.get('/sdapi/v1/options', dependencies=[Depends(_require_api_token)])
def options():
    return {
        'sd_model_checkpoint': _default_checkpoint(),
        'sd_vae': 'None',
    }


@app.get('/sdapi/v1/sd-models', dependencies=[Depends(_require_api_token)])
def sd_models():
    return [{'title': m, 'model_name': m, 'hash': ''} for m in _checkpoint_filenames()]


@app.post('/sdapi/v1/txt2img', dependencies=[Depends(_require_api_token)])
async def txt2img(req: Txt2ImgRequest):
    t0 = time.perf_counter()
    try:
        resp = await asyncio.get_running_loop().run_in_executor(None, _submit_task, req)
    except _BusyError:
        return JSONResponse({'detail': 'generation already in progress'}, status_code=429)
    except Exception as e:
        traceback.print_exc()
        _kill_worker()
        return JSONResponse({'detail': str(e)}, status_code=500)

    if not resp.get('ok'):
        _kill_worker()
        return JSONResponse({'detail': resp.get('error', 'unknown generation error')}, status_code=500)

    images = []
    for p in resp.get('images', []):
        try:
            with open(p, 'rb') as f:
                images.append('data:image/png;base64,' + base64.b64encode(f.read()).decode())
        finally:
            try:
                os.remove(p)
            except Exception:
                pass

    _last_activity = time.time()
    elapsed = time.perf_counter() - t0
    print(f'[Fooosti] generation done in {elapsed:.1f}s', flush=True)

    result = {
        'images': images,
        'info': json.dumps({'prompt': req.prompt}),
    }
    return JSONResponse(result)

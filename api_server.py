import asyncio
import base64
import hmac
import json
import os
import threading
import time
import traceback
import uuid

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

import args_manager
from modules import config, constants, ipc

API_TOKEN = os.environ.get('FOOOSTI_API_TOKEN', '')
GENERATION_TIMEOUT = int(os.environ.get('FOOOSTI_GENERATION_TIMEOUT') or constants.FOOOSTI_GENERATION_TIMEOUT)

if not API_TOKEN:
    print('[Fooosti] WARNING: FOOOSTI_API_TOKEN is not set - /sdapi/v1/* endpoints are unauthenticated.', flush=True)

_current_task_id = None
_current_task_lock = threading.RLock()

_tasks = {}
_tasks_lock = threading.RLock()


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


def _user_config_path():
    return os.environ.get('config_path', os.path.join(os.environ.get('DATADIR', '/content/data'), 'config.txt'))


def _user_config():
    try:
        with open(_user_config_path(), 'r', encoding='utf-8') as f:
            cfg = json.load(f)
            return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def _persist_config(delta: dict):
    import json
    cfg = {}
    try:
        with open(_user_config_path(), 'r', encoding='utf-8') as f:
            cfg = json.load(f)
    except Exception:
        cfg = {}
    if not isinstance(cfg, dict):
        cfg = {}
    cfg.update(delta)
    tmp = _user_config_path() + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=4)
    os.replace(tmp, _user_config_path())


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


def _on_message(msg):
    tid = msg.get('id')
    if not tid:
        return
    with _tasks_lock:
        entry = _tasks.get(tid)
    if entry is None:
        return
    etype = msg.get('type')
    if etype == 'progress':
        entry['progress'] = msg.get('payload', {})
    elif etype == 'results':
        entry['results'] = msg.get('images', [])
    elif etype == 'finish':
        entry['done'] = True
        entry['results'] = msg.get('results', [])
        entry['ev'].set()
    elif etype in ('error', 'cancelled'):
        entry['done'] = True
        entry['error'] = msg.get('message') or ('Generation cancelled' if etype == 'cancelled' else 'Generation failed')
        entry['ev'].set()


ipc.init()
ipc.start_reader(_on_message)


def _submit_task(req: Txt2ImgRequest) -> dict:
    global _current_task_id
    task_id = uuid.uuid4().hex
    task_dict = req.model_dump()
    # resolve the model here from the current config.txt so a change made via
    # POST /sdapi/v1/options applies to the next request even with a warm worker.
    # Never let a raw name escape the checkpoints dir: strip any path components.
    task_dict['base_model_name'] = os.path.basename(task_dict['base_model_name'] or '')
    task_dict['base_model_name'] = task_dict['base_model_name'] or _default_checkpoint()

    # the prompt translator model is downloaded by the main process (mirrors
    # launch_webui.py), never by the worker; make sure it is present before enqueueing
    if config.enable_prompt_translator:
        from modules.prompt_translator import is_available, download
        if not is_available():
            download()

    with _tasks_lock:
        _tasks[task_id] = {'ev': threading.Event(), 'done': False, 'error': None,
                           'results': [], 'progress': {}}
    with _current_task_lock:
        _current_task_id = task_id

    ipc.send({'cmd': 'submit', 'source': 'api', 'id': task_id, 'payload': {'task': task_dict}})

    try:
        with _tasks_lock:
            entry = _tasks[task_id]
        if not entry['ev'].wait(GENERATION_TIMEOUT):
            # the task is still pending (possibly queued behind another source);
            # ask the manager to purge/abort it
            ipc.send({'cmd': 'stop', 'source': 'api'})
            raise TimeoutError('Fooosti generation timed out')
        if entry['error']:
            raise RuntimeError(entry['error'])
        return {'ok': True, 'images': entry['results']}
    finally:
        with _tasks_lock:
            _tasks.pop(task_id, None)
        with _current_task_lock:
            if _current_task_id == task_id:
                _current_task_id = None


app = FastAPI(title='Fooosti', version='0.1.0')

# --- Lazy-idle: track request activity and exit this process when the API is
# unused so the daemon can take the port back. Active in lazy mode only
# (API_KEEPALIVE_MINUTES != -1): with 0 it exits shortly after the last request,
# with N>0 after N minutes without any request. In-flight requests keep it alive
# (a generation may take minutes even though no new request arrived).
_idle_inflight = 0
_idle_last = time.time()


@app.middleware('http')
async def _idle_touch(request, call_next):
    global _idle_last, _idle_inflight
    _idle_inflight += 1
    try:
        return await call_next(request)
    finally:
        _idle_inflight -= 1
        _idle_last = time.time()


def _install_idle_kill():
    try:
        keepalive = float(os.environ.get('API_KEEPALIVE_MINUTES', '-1') or '-1')
    except ValueError:
        keepalive = -1.0
    if keepalive == -1:
        return
    idle_seconds = keepalive * 60 if keepalive > 0 else \
        float(os.environ.get('API_IDLE_SECONDS', '2') or '2')

    def _monitor():
        global _idle_last, _idle_inflight
        while True:
            time.sleep(1)
            if _idle_inflight > 0:
                continue
            if time.time() - _idle_last >= idle_seconds:
                print(f'[api] no requests for {idle_seconds:.0f}s, exiting', flush=True)
                os._exit(0)

    threading.Thread(target=_monitor, daemon=True).start()


_install_idle_kill()


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


@app.post('/sdapi/v1/options', dependencies=[Depends(_require_api_token)])
def set_options(payload: dict):
    """A1111-compatible: accept a full options dict, persist what we support.
    Open WebUI switches the model by POSTing the dict it got from GET."""
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail='options must be a JSON object')
    delta = {}
    if payload.get('sd_model_checkpoint'):
        delta['default_model'] = payload['sd_model_checkpoint']
    if delta:
        try:
            _persist_config(delta)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'failed to persist config: {e}')
    return options()


@app.get('/sdapi/v1/sd-models', dependencies=[Depends(_require_api_token)])
def sd_models():
    return [{'title': m, 'model_name': m, 'hash': ''} for m in _checkpoint_filenames()]


@app.post('/sdapi/v1/interrupt', dependencies=[Depends(_require_api_token)])
def interrupt():
    """Interrupt the in-flight generation (A1111-compatible)."""
    ipc.send({'cmd': 'stop', 'source': 'api'})
    return {}


@app.post('/sdapi/v1/skip', dependencies=[Depends(_require_api_token)])
def skip():
    """Skip the current image of the in-flight generation (A1111-compatible)."""
    ipc.send({'cmd': 'skip', 'source': 'api'})
    return {}


@app.get('/sdapi/v1/progress', dependencies=[Depends(_require_api_token)])
def progress(skip_current_image: bool = False):
    """Live progress of the in-flight generation (A1111-compatible)."""
    with _current_task_lock:
        tid = _current_task_id
    state = {
        'skipped': False,
        'interrupted': False,
        'job': 'txt2img',
        'job_count': 1,
        'job_no': 0,
        'job_timestamp': '',
        'sampling_step': 0,
        'sampling_steps': 0,
    }
    with _tasks_lock:
        entry = _tasks.get(tid) if tid else None
    if entry is None:
        return {'progress': 0.0, 'eta_relative': 0.0, 'state': state, 'current_image': None}
    payload = entry.get('progress', {})
    return {
        'progress': payload.get('progress', 0.0),
        'eta_relative': payload.get('eta_relative', 0.0),
        'state': state,
        'current_image': None if skip_current_image else payload.get('current_image'),
    }


def _temp_sweep():
    # remove scratch images left behind by aborted/killed API tasks
    import glob
    while True:
        time.sleep(300)
        now = time.time()
        try:
            for p in glob.glob(os.path.join(config.temp_path, 'api_*.png')):
                try:
                    if now - os.path.getmtime(p) > 3600:
                        os.remove(p)
                except Exception:
                    pass
        except Exception:
            pass


threading.Thread(target=_temp_sweep, daemon=True).start()


@app.post('/sdapi/v1/txt2img', dependencies=[Depends(_require_api_token)])
async def txt2img(req: Txt2ImgRequest):
    t0 = time.perf_counter()
    try:
        resp = await asyncio.get_running_loop().run_in_executor(None, _submit_task, req)
    except TimeoutError as e:
        print(f'[Fooosti] generation timed out: {e}', flush=True)
        return JSONResponse({'detail': str(e)}, status_code=500)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({'detail': str(e)}, status_code=500)

    if not resp.get('ok'):
        return JSONResponse({'detail': resp.get('error', 'unknown generation error')}, status_code=500)

    images = []
    for p in resp.get('images', []):
        try:
            with open(p, 'rb') as f:
                images.append('data:image/png;base64,' + base64.b64encode(f.read()).decode())
        finally:
            # API tasks run with temp_only in the worker: results are written to
            # its scratch temp dir, never into the WebUI outputs/history, so we
            # always remove them here
            try:
                os.remove(p)
            except Exception:
                pass

    elapsed = time.perf_counter() - t0
    print(f'[Fooosti] generation done in {elapsed:.1f}s', flush=True)

    result = {
        'images': images,
        'parameters': req.model_dump(),
        'info': json.dumps({'prompt': req.prompt}),
    }
    ipc.trim_memory()
    return JSONResponse(result)

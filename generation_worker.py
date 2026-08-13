import json
import os
import sys
import threading
import time
import traceback
import uuid

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
os.chdir(root)


def _set_pdeathsig():
    if os.name != 'posix':
        return
    try:
        import ctypes
        import signal
        libc = ctypes.CDLL('libc.so.6', use_errno=True)
        if libc.prctl(1, signal.SIGKILL) != 0:  # PR_SET_PDEATHSIG
            raise OSError(ctypes.get_errno(), 'prctl(PR_SET_PDEATHSIG) failed')
        # if the parent died before we set the flag, exit now
        if os.getppid() == 1:
            os.kill(os.getpid(), signal.SIGKILL)
    except Exception as e:
        print(f'[Fooosti] PDEATHSIG unavailable: {e}', flush=True)


_set_pdeathsig()

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

from args_manager import args
from modules import config

if args.gpu_device_id is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_device_id)
    print("Set device to:", args.gpu_device_id)

if args.hf_mirror is not None:
    os.environ['HF_MIRROR'] = str(args.hf_mirror)
    print("Set hf_mirror to:", args.hf_mirror)

os.environ["U2NET_HOME"] = config.path_inpaint
os.environ['GRADIO_TEMP_DIR'] = config.temp_path

vae_approx_filenames = [
    ('xlvaeapp.pth', 'https://huggingface.co/lllyasviel/misc/resolve/main/xlvaeapp.pth'),
    ('vaeapp_sd15.pth', 'https://huggingface.co/lllyasviel/misc/resolve/main/vaeapp_sd15.pt'),
    ('xl-to-v1_interposer-v4.0.safetensors',
     'https://huggingface.co/mashb1t/misc/resolve/main/xl-to-v1_interposer-v4.0.safetensors')
]

from modules.model_loader import load_file_from_url
try:
    for file_name, url in vae_approx_filenames:
        load_file_from_url(url=url, model_dir=config.path_vae_approx, file_name=file_name)
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin',
        model_dir=config.path_fooocus_expansion,
        file_name='pytorch_model.bin'
    )
except Exception as e:
    print(f'[Fooosti] warning: optional model download check failed: {e}', flush=True)

config.update_files()

import modules.memory
keepalive_minutes = int(os.environ.get('FOOOSTI_KEEPALIVE_MINUTES', '0') or '0')
modules.memory.KEEP_WARM = keepalive_minutes > 0
print(f'[Fooosti] worker keepalive={keepalive_minutes}min')

import modules.async_worker  # noqa: E402  (starts worker thread + patch_all)
from modules import constants

OUT_DIR = config.path_outputs
os.makedirs(OUT_DIR, exist_ok=True)

# stop/interrupt control channel, shared with modules/remote_worker.py
STOP_DIR = os.environ.get('FOOOSTI_TMP_DIR', constants.FOOOSTI_TMP_DIR)
os.makedirs(STOP_DIR, exist_ok=True)

# True while a generation is in progress; SIGTERM (parent kill) must not touch
# torch state mid-sampling, so release_all is skipped in that case
_busy_generating = False


def build_task_args(req: dict) -> list:
    import modules.flags
    import modules.sdxl_styles
    from modules import constants
    from modules.flags import Performance

    default_max_lora_number = config.default_max_lora_number

    style_selections = req.get('style_selections')
    if style_selections is None:
        style_selections = config.default_styles
    style_selections = [s for s in style_selections if s in modules.sdxl_styles.legal_style_names]

    performance = req.get('performance')
    if performance not in Performance.values():
        performance = config.default_performance
    perf = Performance(performance)

    width, height = req.get('width'), req.get('height')
    aspect_ratios_selection = config.default_aspect_ratio
    if (width is None) != (height is None):
        raise ValueError('Provide both width and height (or neither)')
    if width is not None and height is not None:
        aspect_ratios_selection = f"{width}×{height}"

    seed = req.get('seed', -1)
    if seed is None or seed < 0:
        import random
        seed = random.randint(0, constants.MAX_SEED)
    seed = int(seed)

    lora_ctrls = []
    for _ in range(default_max_lora_number):
        lora_ctrls += [False, 'None', 0.5]

    ip_ctrls = []
    for _ in range(config.default_controlnet_image_count):
        ip_ctrls += [None, 0.8, 1.0, 'PyraCanny']

    enhance_ctrls = []
    for _ in range(config.default_enhance_tabs):
        enhance_ctrls += [
            False, '', '', '', config.default_enhance_inpaint_mask_model,
            config.default_inpaint_mask_cloth_category, config.default_inpaint_mask_sam_model,
            0.25, 0.3, config.default_sam_max_detections,
            False, config.default_inpaint_engine_version, 1.0, 0.618, 0, False,
        ]

    inpaint_engine = config.default_inpaint_engine_version
    if perf in [Performance.LIGHTNING, Performance.HYPER_SD, Performance.EXTREME_SPEED]:
        inpaint_engine = 'None'

    steps = req.get('steps')
    if steps is not None:
        steps = min(max(int(steps), 1), 200)
    overwrite_step = steps if steps is not None else config.default_overwrite_step
    overwrite_width = width if width else -1
    overwrite_height = height if height else -1

    batch_size = req.get('batch_size')
    batch_size = batch_size if batch_size and batch_size > 0 else 1

    metadata_scheme = req.get('metadata_scheme')
    if not metadata_scheme:
        metadata_scheme = config.default_metadata_scheme

    task_args = [
        False,
        req.get('prompt', ''),
        req.get('negative_prompt') if req.get('negative_prompt') is not None else config.default_prompt_negative,
        style_selections,
        performance,
        aspect_ratios_selection,
        batch_size,
        'png',
        seed,
        False,
        req.get('sharpness') if req.get('sharpness') is not None else config.default_sample_sharpness,
        req.get('cfg_scale') if req.get('cfg_scale') is not None else config.default_cfg_scale,
        req.get('base_model_name') if req.get('base_model_name') else config.default_base_model_name,
        config.default_refiner_model_name,
        config.default_refiner_switch,
    ] + lora_ctrls + [
        config.default_image_prompt_checkbox,
        'uov',
        config.default_uov_method,
        None,
        [], None, '', None,
        True, True, False, config.default_black_out_nsfw,
        1.5, 0.8, 0.3,
        config.default_cfg_tsnr,
        config.default_clip_skip,
        req.get('sampler_name') if req.get('sampler_name') else config.default_sampler,
        req.get('scheduler_name') if req.get('scheduler_name') else config.default_scheduler,
        config.default_vae,
        overwrite_step,
        config.default_overwrite_switch,
        overwrite_width,
        overwrite_height,
        -1,
        config.default_overwrite_upscale,
        False, False,
        False, False, 1, 255,
        modules.flags.refiner_swap_method,
        0.25,
        False, 1.01, 1.02, 0.99, 0.95,
        False, False, inpaint_engine, 1.0, 0.618,
        config.default_inpaint_advanced_masking_checkbox,
        config.default_invert_mask_checkbox,
        0,
    ]
    # These three positional args are conditionally appended by the UI
    # (webui.py) and must match AsyncTask.__init__'s conditional pops, so the
    # worker flags must be mirrored here or the arg alignment breaks.
    if not args.disable_image_log:
        task_args.append(config.default_save_only_final_enhanced_image)
    if not args.disable_metadata:
        task_args += [config.default_save_metadata_to_images, metadata_scheme]

    task_args += ip_ctrls + [
        False, 0, False,
        None, False, config.default_enhance_uov_method,
        config.default_enhance_uov_processing_order,
        config.default_enhance_uov_prompt_type,
    ] + enhance_ctrls

    expected = 70
    if not args.disable_image_log:
        expected += 1
    if not args.disable_metadata:
        expected += 2
    expected += (3 * default_max_lora_number
                 + 4 * config.default_controlnet_image_count
                 + 16 * config.default_enhance_tabs)
    if len(task_args) != expected:
        raise RuntimeError(f'worker arg count mismatch: built {len(task_args)}, expected {expected}')
    return task_args


def _poll_interval():
    # sub-second so Stop/Skip/interrupt land within one diffusion step boundary
    return 0.05


def run_task(task_dict: dict, resp_file: str, task_id: str = '', progress_file: str = ''):
    global _busy_generating
    import numpy as np
    from PIL import Image
    from modules import async_worker
    from modules.async_worker import AsyncTask
    import ldm_patched.modules.model_management as model_management

    resp = {'ok': False}
    done = threading.Event()
    task = None

    def watcher():
        stop_file = os.path.join(STOP_DIR, f'stop_{task_id}')
        while not done.is_set():
            if os.path.exists(stop_file):
                try:
                    with open(stop_file) as f:
                        value = f.read().strip() or 'stop'
                except Exception:
                    value = 'stop'
                try:
                    os.remove(stop_file)
                except Exception:
                    pass
                if task is not None:
                    task.last_stop = value
                model_management.interrupt_current_processing()
                break
            time.sleep(_poll_interval())

    def progress_reporter():
        t0 = time.perf_counter()
        last_img_t = 0.0
        seen = 0
        while not done.is_set():
            # scan only newly-appended yields (avoids the O(n^2) re-scan)
            while seen < len(task.yields):
                item = task.yields[seen]
                seen += 1
                if item[0] == 'preview':
                    pct, _text, img = item[1]
                    entry = {
                        'progress': min(max(float(pct), 0.0), 100.0) / 100.0,
                        'eta_relative': 0.0,
                        'current_image': None,
                    }
                    now = time.perf_counter()
                    if img is not None and now - last_img_t >= 1.0:
                        last_img_t = now
                        entry['current_image'] = _encode_preview(img)
                    p = entry['progress']
                    elapsed = now - t0
                    if p > 0.01:
                        entry['eta_relative'] = elapsed / p * (1.0 - p)
                    _atomic_write(progress_file, json.dumps(entry))
            time.sleep(_poll_interval())

    try:
        t0 = time.perf_counter()
        args_list = build_task_args(task_dict)
        task = AsyncTask(args=args_list)

        # clear a leftover interrupt flag before the watcher can observe anything
        model_management.interrupt_current_processing(False)

        if task_id:
            threading.Thread(target=watcher, daemon=True).start()
        if progress_file:
            threading.Thread(target=progress_reporter, daemon=True).start()

        _busy_generating = True
        async_worker.async_tasks.append(task)

        deadline = time.time() + float(os.environ.get('FOOOSTI_GENERATION_TIMEOUT') or str(constants.FOOOSTI_GENERATION_TIMEOUT))
        error = None
        done_polling = False
        seen = 0
        while time.time() < deadline:
            while seen < len(task.yields):
                item = task.yields[seen]
                seen += 1
                if item[0] == 'finish':
                    done_polling = True
                    break
                if item[0] == 'error':
                    error = item[1] if len(item) > 1 else 'Generation failed'
                    done_polling = True
                    break
            if done_polling:
                break
            time.sleep(_poll_interval())
        else:
            # interrupt the worker so the orphaned generation settles quickly
            # instead of running to completion behind the next request
            model_management.interrupt_current_processing(True)
            raise TimeoutError('Fooosti generation timed out')

        if error is not None:
            raise RuntimeError(str(error))

        images = []
        for i, item in enumerate(task.results):
            if isinstance(item, str):
                images.append(item)
            elif isinstance(item, np.ndarray):
                p = os.path.join(config.temp_path, f'api_{uuid.uuid4().hex[:8]}_{i}.png')
                os.makedirs(config.temp_path, exist_ok=True)
                Image.fromarray(item.astype(np.uint8)).save(p, format='PNG')
                images.append(p)
            else:
                continue

        resp = {'ok': True, 'images': images, 'elapsed': round(time.perf_counter() - t0, 1)}
    except Exception as e:
        traceback.print_exc()
        resp = {'ok': False, 'error': str(e)}
    finally:
        _busy_generating = False
        done.set()

    _atomic_write(resp_file, json.dumps(resp))
    return resp


def _atomic_write(path, data):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        f.write(data)
    os.replace(tmp, path)


def _encode_preview(img, max_dim=512):
    import base64
    import cv2
    import numpy as np
    try:
        h, w = img.shape[:2]
        scale = min(1.0, max_dim / max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            return base64.b64encode(buf.tobytes()).decode()
    except Exception:
        pass
    return None


def _deserialize_value(a):
    import base64
    import cv2
    import numpy as np
    if isinstance(a, dict):
        if a.get('__ndarray__'):
            buf = np.frombuffer(base64.b64decode(a['__ndarray__']), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
        return {k: _deserialize_value(v) for k, v in a.items()}
    if isinstance(a, list):
        return [_deserialize_value(x) for x in a]
    if isinstance(a, tuple):
        return tuple(_deserialize_value(x) for x in a)
    return a


def _deserialize_args(args):
    return [_deserialize_value(a) for a in args]


def run_task_stream(task_id: str, args_list: list):
    import base64
    import cv2
    import numpy as np
    from modules import async_worker
    from modules.async_worker import AsyncTask

    stop_file = os.path.join(STOP_DIR, f'stop_{task_id}')
    done = threading.Event()

    def emit(etype, **kw):
        print(json.dumps({'kind': 'event', 'id': task_id, 'type': etype, **kw}), flush=True)

    def watcher():
        while not done.is_set():
            if os.path.exists(stop_file):
                try:
                    with open(stop_file) as f:
                        value = f.read().strip() or 'stop'
                except Exception:
                    value = 'stop'
                try:
                    os.remove(stop_file)
                except Exception:
                    pass
                task.last_stop = value
                import ldm_patched.modules.model_management
                ldm_patched.modules.model_management.interrupt_current_processing()
                break
            time.sleep(_poll_interval())

    timeout = float(os.environ.get('FOOOSTI_GENERATION_TIMEOUT') or str(constants.FOOOSTI_GENERATION_TIMEOUT))
    deadline = time.time() + timeout

    try:
        task = AsyncTask(args=_deserialize_args(args_list))

        # clear a leftover interrupt flag so it cannot abort this fresh task
        import ldm_patched.modules.model_management
        ldm_patched.modules.model_management.interrupt_current_processing(False)

        threading.Thread(target=watcher, daemon=True).start()

        global _busy_generating
        _busy_generating = True
        async_worker.async_tasks.append(task)

        while True:
            if time.time() > deadline:
                ldm_patched.modules.model_management.interrupt_current_processing(True)
                raise TimeoutError(f'Fooosti generation timed out ({timeout}s)')
            if task.yields:
                flag, product = task.yields.pop(0)
                if flag == 'preview':
                    pct, title, img = product
                    img_b64 = None
                    if isinstance(img, np.ndarray):
                        ok, buf = cv2.imencode('.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                        if ok:
                            img_b64 = base64.b64encode(buf.tobytes()).decode()
                    emit('preview', payload=[pct, title, img_b64])
                elif flag == 'results':
                    emit('results', images=[p for p in product if isinstance(p, str)])
                elif flag == 'error':
                    emit('error', message=product if isinstance(product, str) else 'Generation failed',
                         results=[])
                    return
                elif flag == 'finish':
                    emit('finish',
                         results=[p for p in product if isinstance(p, str)],
                         should_enhance=task.should_enhance,
                         enhance_stats=task.enhance_stats,
                         images_to_enhance_count=task.images_to_enhance_count)
                    return
            else:
                time.sleep(0.1)
    except Exception as e:
        traceback.print_exc()
        try:
            emit('error', message=str(e), results=[])
        except Exception:
            pass
    finally:
        _busy_generating = False
        done.set()


def _sigterm_handler(signum, frame):
    # the parent kills us when keepalive expires (or after a task when
    # keepalive=0); free VRAM/RAM explicitly rather than relying on process exit
    try:
        if not _busy_generating:
            import modules.memory
            modules.memory.release_all(force=True)
    except Exception:
        pass
    os._exit(0)


def main():
    import signal
    signal.signal(signal.SIGTERM, _sigterm_handler)
    print('[Fooosti] worker ready, waiting for tasks', flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
            if msg.get('kind') == 'task':
                print(f"[Fooosti] running task {msg.get('id')} ...", flush=True)
                run_task_stream(msg.get('id'), msg.get('args', []))
                print(f"[Fooosti] task {msg.get('id')} done", flush=True)
                continue
            task_dict = msg.get('task', {})
            resp_file = msg.get('resp_file')
            if not resp_file:
                continue
            print(f"[Fooosti] running task ...", flush=True)
            run_task(task_dict, resp_file,
                     task_id=msg.get('id') or '',
                     progress_file=msg.get('progress_file') or '')
            print(f"[Fooosti] task done -> {resp_file}", flush=True)
        except Exception as e:
            traceback.print_exc()
            print(f'[Fooosti] bad message: {e}', flush=True)
    print('[Fooosti] worker exiting', flush=True)


if __name__ == '__main__':
    main()

import os
import sys
import threading
import time
import traceback
import uuid

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
os.chdir(root)

from modules import ipc
ipc.init()


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
from modules import config, constants

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
KEEPALIVE_MINUTES = int(os.environ.get('WORKER_KEEPALIVE_MINUTES', '0') or '0')
modules.memory.KEEP_WARM = KEEPALIVE_MINUTES > 0
print(f'[Fooosti] worker keepalive={KEEPALIVE_MINUTES}min')

import modules.async_worker  # noqa: E402  (starts worker thread + patch_all)

OUT_DIR = config.path_outputs
os.makedirs(OUT_DIR, exist_ok=True)

# True while a generation is in progress; SIGTERM (manager kill) must not touch
# torch state mid-sampling, so release_all is skipped in that case
_busy_generating = False

_state_lock = threading.RLock()
_current = None  # {'source': ..., 'id': ..., 'task': AsyncTask}
_pending_interrupt = None  # 'stop'|'skip' seen before the task was created
_shutdown = threading.Event()
_task_queue = []


def _emit(task_id, etype, **kw):
    ipc.send({'type': 'event', 'id': task_id, 'event': {'id': task_id, 'type': etype, **kw}})


def _emit_preview(task_id, product, state):
    import base64
    import cv2
    import numpy as np
    pct, title, img = product
    img_b64 = None
    if isinstance(img, np.ndarray):
        try:
            ok, buf = cv2.imencode('.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            if ok:
                img_b64 = base64.b64encode(buf.tobytes()).decode()
        except Exception:
            pass
    _emit(task_id, 'preview', payload=[pct, title, img_b64])

    now = time.perf_counter()
    current_image = None
    if img_b64 is not None and now - state['last_img'] >= 1.0:
        current_image = img_b64
        state['last_img'] = now
    p = min(max(float(pct), 0.0), 100.0) / 100.0
    elapsed = now - state['t0']
    eta_relative = 0.0
    if p > 0.01:
        eta_relative = elapsed / p * (1.0 - p)
    _emit(task_id, 'progress',
          payload={'progress': p, 'eta_relative': eta_relative, 'current_image': current_image})


def _emit_finish(task_id, task, source):
    import numpy as np
    from PIL import Image
    results = []
    for item in task.results:
        if isinstance(item, str):
            results.append(item)
        elif isinstance(item, np.ndarray) and source == 'api':
            try:
                p = os.path.join(config.temp_path, f'api_{uuid.uuid4().hex[:8]}_{len(results)}.png')
                os.makedirs(config.temp_path, exist_ok=True)
                Image.fromarray(item.astype(np.uint8)).save(p, format='PNG')
                results.append(p)
            except Exception:
                pass
    _emit(task_id, 'finish',
          results=results,
          should_enhance=task.should_enhance,
          enhance_stats=task.enhance_stats,
          images_to_enhance_count=task.images_to_enhance_count)


def _interrupt(current, value):
    task = current.get('task')
    if task is not None:
        task.last_stop = value
    try:
        import ldm_patched.modules.model_management
        ldm_patched.modules.model_management.interrupt_current_processing()
    except Exception:
        pass


def _sweep_temp():
    import glob
    now = time.time()
    for p in glob.glob(os.path.join(config.temp_path, 'api_*.png')):
        try:
            if now - os.path.getmtime(p) > 3600:
                os.remove(p)
        except Exception:
            pass


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


def run_task_data(task_data: dict):
    global _busy_generating, _current, _pending_interrupt
    from modules import async_worker, private_logger
    from modules.async_worker import AsyncTask
    import ldm_patched.modules.model_management as model_management

    source = task_data.get('source', 'webui')
    task_id = task_data.get('id') or uuid.uuid4().hex

    try:
        kind, content = ipc.extract_task_payload(task_data)
        if kind == 'args':
            task = AsyncTask(args=ipc._deserialize_args(content))
        else:
            task = AsyncTask(args=build_task_args(content))
    except Exception as e:
        traceback.print_exc()
        _emit(task_id, 'error', message=str(e), results=[])
        return

    # clear a leftover interrupt flag before the new task can observe anything
    model_management.interrupt_current_processing(False)

    t0 = time.perf_counter()
    state = {'t0': t0, 'last_img': 0.0}
    timeout = float(os.environ.get('FOOOSTI_GENERATION_TIMEOUT') or str(constants.FOOOSTI_GENERATION_TIMEOUT))
    deadline = time.time() + timeout
    error = None

    with _state_lock:
        _current = {'source': source, 'id': task_id, 'task': task}
    _busy_generating = True
    with _state_lock:
        pending = _pending_interrupt
        _pending_interrupt = None
    if pending is not None and _current is not None:
        _interrupt(_current, pending)
    # API tasks must not pollute the WebUI outputs/history: images go to the
    # scratch temp dir instead, where the API server removes them after reading
    if source == 'api':
        private_logger.temp_only = True

    try:
        async_worker.async_tasks.append(task)
        while True:
            if time.time() > deadline:
                # interrupt the worker so the orphaned generation settles
                # quickly instead of running to completion behind the next task
                model_management.interrupt_current_processing(True)
                raise TimeoutError(f'Fooosti generation timed out ({timeout}s)')
            if task.yields:
                flag, product = task.yields.pop(0)
                if flag == 'preview':
                    _emit_preview(task_id, product, state)
                elif flag == 'results':
                    _emit(task_id, 'results', images=[p for p in product if isinstance(p, str)])
                elif flag == 'error':
                    error = product if isinstance(product, str) else 'Generation failed'
                    break
                elif flag == 'finish':
                    _emit_finish(task_id, task, source)
                    break
            else:
                time.sleep(0.1)
    except Exception as e:
        traceback.print_exc()
        error = str(e)
    finally:
        private_logger.temp_only = False
        _busy_generating = False
        with _state_lock:
            _current = None

    if error is not None:
        _emit(task_id, 'error', message=error, results=[])


def _sigterm_handler(signum, frame):
    # external shutdown signal (docker stop, manager kill): free VRAM/RAM
    # explicitly rather than relying on process exit
    try:
        if not _busy_generating:
            import modules.memory
            modules.memory.release_all(force=True)
    except Exception:
        pass
    os._exit(0)


def _on_message(msg):
    cmd = msg.get('cmd')
    if cmd == 'task':
        _task_queue.append({'id': msg.get('id'), 'source': msg.get('source', 'webui'),
                            'payload': msg.get('payload')})
    elif cmd == 'control':
        action = msg.get('action')
        if action in ('stop', 'skip'):
            with _state_lock:
                current = _current
                if current is None:
                    # the task has not been created yet (worker starting up or
                    # still between tasks): remember the interrupt so it applies
                    # as soon as the next task starts, instead of being dropped
                    _pending_interrupt = action
            if current is not None:
                _interrupt(current, action)


def main():
    import signal
    signal.signal(signal.SIGTERM, _sigterm_handler)
    ipc.start_reader(_on_message)
    _sweep_temp()
    ipc.send({'type': 'ready'})
    print('[Fooosti] worker ready, waiting for tasks', flush=True)
    while not _shutdown.is_set():
        if _task_queue:
            task_data = _task_queue.pop(0)
            print(f"[Fooosti] running task {task_data.get('id')} ({task_data.get('source')}) ...", flush=True)
            try:
                run_task_data(task_data)
            except Exception:
                traceback.print_exc()
            ipc.send({'type': 'task_done', 'id': task_data.get('id')})
            print(f"[Fooosti] task {task_data.get('id')} done", flush=True)
        else:
            time.sleep(0.1)
    print('[Fooosti] worker exiting', flush=True)


if __name__ == '__main__':
    main()

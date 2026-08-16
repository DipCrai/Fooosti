KEEP_WARM = False
_keep_warm_warned = False


def rss_mb():
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS'):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return -1


def release_all(force=False):
    global _keep_warm_warned
    if KEEP_WARM and not force:
        if not _keep_warm_warned:
            _keep_warm_warned = True
            print('[Fooosti] WARNING: WORKER_KEEPALIVE_MINUTES>0 keeps the worker warm, '
                  '"memory free after each generation" is disabled', flush=True)
        return

    import gc
    import ctypes
    import torch
    import modules.default_pipeline as default_pipeline
    import modules.sample_hijack
    import modules.upscaler
    import ldm_patched.modules.model_management as model_management
    from modules import core

    before = rss_mb()

    default_pipeline.model_base = core.StableDiffusionModel()
    default_pipeline.model_refiner = core.StableDiffusionModel()
    default_pipeline.final_expansion = None
    default_pipeline.final_unet = None
    default_pipeline.final_clip = None
    default_pipeline.final_vae = None
    default_pipeline.final_refiner_unet = None
    default_pipeline.final_refiner_vae = None
    default_pipeline.loaded_ControlNets = {}
    modules.sample_hijack.current_refiner = None
    modules.upscaler.model = None
    core.VAE_approx_models = {}

    model_management.unload_all_models()
    model_management.soft_empty_cache(force=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        libc = ctypes.CDLL('libc.so.6')
        libc.malloc_trim(0)
    except Exception:
        pass

    after = rss_mb()
    print(f'[Fooosti] released: VRAM+RAM (rss {before}MB -> {after}MB)')

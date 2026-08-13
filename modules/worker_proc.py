"""Shared generation-worker process management.

The WebUI (modules/remote_worker.py) and the API server (api_server.py) both
spawn generation_worker.py and kill it with the same close-stdin +
terminate/wait/kill sequence. Keeping that here means the two callers only
describe their protocol (stdout handling, extra flags), not the process
boilerplate.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER_SCRIPT = os.path.join(ROOT, 'generation_worker.py')


def spawn(*extra_args, stdout=None):
    """Launch a worker, forwarding this process's CLI args so
    --gpu-device-id/--preset/... reach it too."""
    cmd = [sys.executable, WORKER_SCRIPT] + sys.argv[1:] + list(extra_args)
    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=stdout,
        cwd=ROOT,
    )


def terminate(proc):
    """Close stdin, then terminate and wait up to 10s, falling back to kill."""
    if proc is None or proc.poll() is not None:
        return False
    try:
        proc.stdin.close()
    except Exception:
        pass
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    return True


def alive(proc):
    return proc is not None and proc.poll() is None


def keepalive_reaper(name, proc_getter, is_busy, idle_seconds, kill_proc):
    """Daemon loop: kill the worker when it has been idle for longer than
    KEEPALIVE_MINUTES. The caller supplies accessors for the current worker,
    the in-flight signal and the idle time; this keeps the policy in one
    place while the two servers track their own state."""
    import time
    minutes = int(os.environ.get('FOOOSTI_KEEPALIVE_MINUTES', '0') or '0')
    while True:
        time.sleep(30)
        if minutes <= 0:
            continue
        if not alive(proc_getter()):
            continue
        if is_busy():
            continue
        idle = idle_seconds()
        if idle > minutes * 60:
            print(f'[{name}] keepalive expired ({idle:.0f}s idle), killing worker', flush=True)
            kill_proc()

import json
import os
import subprocess
import threading
import time
import uuid

from modules import constants, worker_proc

OUT_DIR = os.environ.get('FOOOSTI_TMP_DIR', constants.FOOOSTI_TMP_DIR)
KEEPALIVE_MINUTES = int(os.environ.get('FOOOSTI_KEEPALIVE_MINUTES', '0') or '0')

os.makedirs(OUT_DIR, exist_ok=True)


class AsyncTask:
    def __init__(self, args):
        self.args = list(args)
        self.yields = []
        self.results = []
        self.last_stop = False
        self.processing = False
        self.should_enhance = False
        self.enhance_stats = {}
        self.images_to_enhance_count = 0
        self._id = None
        self._proc_token = None


_worker_proc = None
_events_thread = None
_tasks = {}
_gen = 0
_last_activity = time.time()
_lock = threading.RLock()


def _interrupt_file(task_id):
    return os.path.join(OUT_DIR, f'stop_{task_id}')


def _spawn():
    global _worker_proc, _events_thread, _gen
    _gen += 1
    gen = _gen
    _worker_proc = worker_proc.spawn(stdout=subprocess.PIPE)
    _events_thread = threading.Thread(target=_reader, args=(gen, _worker_proc), daemon=True)
    _events_thread.start()
    print(f'[remote_worker] spawned pid={_worker_proc.pid}', flush=True)


def _cleanup_task(task):
    stop_file = _interrupt_file(task._id)
    try:
        if os.path.exists(stop_file):
            os.remove(stop_file)
    except Exception:
        pass
    _tasks.pop(task._id, None)


def _reader(gen, proc):
    try:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ev.get('kind') != 'event':
                continue
            try:
                _dispatch(ev)
            except Exception as e:
                print(f'[remote_worker] bad event {ev.get("type")}: {e}', flush=True)
    finally:
        # this worker process exited: fail only tasks that were submitted to it
        for tid, task in list(_tasks.items()):
            if task._proc_token == gen and task.processing:
                task.processing = False
                task.yields.append(['error', 'Worker exited unexpectedly'])
                _cleanup_task(task)


def _maybe_kill_after_finish():
    with _lock:
        if KEEPALIVE_MINUTES <= 0 and not _tasks:
            _kill()


def _dispatch(ev):
    task = _tasks.get(ev.get('id'))
    if task is None:
        return
    etype = ev.get('type')
    if etype == 'preview':
        pct, title, img_b64 = ev['payload']
        img = None
        if img_b64:
            import base64
            import cv2
            import numpy as np
            buf = np.frombuffer(base64.b64decode(img_b64), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        task.yields.append(['preview', (pct, title, img)])
    elif etype == 'results':
        task.yields.append(['results', ev['images']])
    elif etype == 'error':
        task.processing = False
        task.yields.append(['error', ev.get('message') or 'Generation failed'])
        _cleanup_task(task)
        _last_activity = time.time()
        _maybe_kill_after_finish()
    elif etype == 'finish':
        task.should_enhance = ev.get('should_enhance', False)
        task.enhance_stats = ev.get('enhance_stats', {})
        task.images_to_enhance_count = ev.get('images_to_enhance_count', 0)
        task.processing = False
        task.yields.append(['finish', ev.get('results', [])])
        _cleanup_task(task)
        _last_activity = time.time()
        _maybe_kill_after_finish()


def _ensure_worker():
    with _lock:
        if _worker_proc is None or _worker_proc.poll() is not None:
            _spawn()


def _kill():
    global _worker_proc, _events_thread
    with _lock:
        if worker_proc.terminate(_worker_proc):
            print(f'[remote_worker] pid={_worker_proc.pid} terminated', flush=True)
        _worker_proc = None
        _events_thread = None


def _guarded_write(proc, line, timeout=60):
    ok = [False]
    err = []

    def _do_write():
        try:
            proc.stdin.write(line)
            proc.stdin.flush()
            ok[0] = True
        except Exception as e:
            err.append(e)

    t = threading.Thread(target=_do_write, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return False, TimeoutError('worker stdin busy')
    if err:
        return False, err[0]
    return True, None


def _serialize_value(a):
    import base64
    import cv2
    import numpy as np
    if a is None or isinstance(a, (str, int, float, bool)):
        return a
    if isinstance(a, np.ndarray):
        arr = a
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            raise ValueError(f'unsupported array for worker IPC: shape={a.shape} dtype={a.dtype}')
        if arr.dtype != np.uint8:
            if arr.dtype in (np.float32, np.float64):
                _mx = float(arr.max())
                if _mx <= 1.0:
                    arr = arr * 255.0
                elif _mx <= 255.0:
                    # float maps in pixel range (e.g. a 0..2 mask): normalize so
                    # the max becomes white instead of a near-black cast
                    arr = arr * (255.0 / _mx)
                arr = np.clip(arr, 0, 255)
            arr = arr.astype(np.uint8)
        if arr.shape[2] == 4:
            # Fooocus only consumes 3-channel BGR images (masks are flattened to
            # grayscale downstream), so an RGBA input is explicitly reduced here
            # rather than silently producing a wrong 3-channel image.
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        else:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode('.png', arr)
        if not ok:
            raise ValueError(f'could not encode image for worker IPC: shape={a.shape} dtype={a.dtype}')
        return {'__ndarray__': base64.b64encode(buf.tobytes()).decode()}
    if isinstance(a, np.integer):
        return int(a)
    if isinstance(a, np.floating):
        return float(a)
    if isinstance(a, np.bool_):
        return bool(a)
    if isinstance(a, (list, tuple)):
        return [_serialize_value(x) for x in a]
    if isinstance(a, dict):
        return {k: _serialize_value(v) for k, v in a.items()}
    return str(a)


def _serialize_args(args):
    return [_serialize_value(a) for a in args]


def _submit(task):
    if task.args is None or len(task.args) == 0:
        return
    task._id = uuid.uuid4().hex
    payload = {'kind': 'task', 'id': task._id, 'args': _serialize_args(task.args)}
    line = (json.dumps(payload) + '\n').encode()
    task.processing = True
    task._proc_token = None
    _tasks[task._id] = task

    with _lock:
        _ensure_worker()
        for attempt in range(2):
            ok, err = _guarded_write(_worker_proc, line)
            if ok:
                task._proc_token = _gen
                _last_activity = time.time()
                return
            _kill()
            _ensure_worker()
        task.processing = False
        task.yields.append(['error', f'Failed to submit task: {err}'])
        _cleanup_task(task)
        _kill()  # the respawned idle worker must not linger (keepalive=0)


class _TaskList(list):
    def append(self, task):
        list.append(self, task)
        _submit(task)


async_tasks = _TaskList()


def request_interrupt(value):
    """Write an explicit interrupt value ('stop' or 'skip') for every running
    task. Unlike mutating the frontend's gr.State copy, this targets the live
    worker tasks directly, so Stop/Skip buttons work regardless of gr.State
    serialization."""
    with _lock:
        for tid, task in list(_tasks.items()):
            if task.processing:
                task.last_stop = value
                with open(_interrupt_file(tid), 'w') as f:
                    f.write(str(value))


def kill_worker():
    with _lock:
        _kill()


def _reaper_busy():
    with _lock:
        return any(t.processing for t in _tasks.values())


threading.Thread(
    target=worker_proc.keepalive_reaper,
    args=('remote_worker',
          lambda: _worker_proc,
          _reaper_busy,
          lambda: time.time() - _last_activity,
          _kill),
    daemon=True).start()

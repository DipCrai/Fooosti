"""Stdio JSON-lines IPC for queue-manager children.

Every child of queue_manager.py (webui/api clients and the generation worker)
talks to the manager over its own stdin/stdout: messages from the manager are
read on stdin, messages to the manager are written on stdout. Child log output
is redirected to stderr so it can never corrupt the protocol.
"""

import json
import sys
import threading

_IN = None
_OUT = None
_lock = threading.Lock()


def init():
    """Take over stdin/stdout as the IPC channel; route logging to stderr.

    Must be called before any other output is produced by the child. Returns
    (in_stream, out_stream). Idempotent: a second call (e.g. api_server being
    imported after launch_api already init'ed) must not recapture the already
    redirected sys.stdout, which would silently route sends to stderr."""
    global _IN, _OUT
    if _OUT is None:
        _OUT = sys.stdout.buffer
        sys.stdout = sys.stderr
    if _IN is None:
        _IN = sys.stdin.buffer
    return _IN, _OUT


def send(msg):
    """Write one JSON message to the manager (thread-safe)."""
    global _OUT
    if _OUT is None:
        init()
    line = (json.dumps(msg) + '\n').encode('utf-8')
    with _lock:
        _OUT.write(line)
        _OUT.flush()


def read_messages():
    """Blocking iterator over stdin; yields parsed messages, stops on EOF."""
    global _IN
    if _IN is None:
        init()
    while True:
        line = _IN.readline()
        if not line:
            return
        line = line.decode('utf-8', 'replace').strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            # garbage on the wire (a C library wrote straight to fd 1):
            # never let it kill the child, just skip it
            continue
        yield msg


def start_reader(handler):
    """Start a daemon thread that feeds stdin messages to `handler(msg)`."""
    def _run():
        try:
            for msg in read_messages():
                try:
                    handler(msg)
                except Exception:
                    pass
        except Exception:
            pass

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


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
                if float(arr.max()) <= 1.0:
                    arr = arr * 255.0
                arr = np.clip(arr, 0, 255)
            arr = arr.astype(np.uint8)
        if arr.shape[2] == 4:
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


def extract_task_payload(task_data: dict):
    """Unpack the client payload the manager forwarded verbatim to the worker.
    api submits payload={'task': {...}}, webui payload={'args': [...]}.
    Returns (kind, content) where kind is 'args' or 'task'."""
    payload = task_data.get('payload') or {}
    if 'args' in payload:
        return 'args', payload['args']
    return 'task', payload.get('task') or {}

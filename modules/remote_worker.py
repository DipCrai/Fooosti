"""WebUI-side client of the fooosti daemon.

Keeps the interface webui.py already uses (AsyncTask, async_tasks,
request_interrupt, kill_worker) but replaces the file-backed queue with
stdio JSON-lines IPC to fooosti.py (see modules/ipc.py). Events from the
worker arrive on stdin and are routed to the matching AsyncTask.
"""

import threading
import uuid

from modules import ipc


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


_tasks = {}
_lock = threading.RLock()


def _cleanup_task(task):
    _tasks.pop(task._id, None)


def _decode_b64(img_b64):
    import base64
    import cv2
    import numpy as np
    try:
        buf = np.frombuffer(base64.b64decode(img_b64), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
    except Exception:
        return None


def _dispatch(task, ev):
    etype = ev.get('type')
    if etype == 'preview':
        payload = ev.get('payload') or []
        pct = payload[0] if len(payload) > 0 else 0
        title = payload[1] if len(payload) > 1 else ''
        img = _decode_b64(payload[2]) if len(payload) > 2 and payload[2] else None
        task.yields.append(['preview', (pct, title, img)])
    elif etype == 'results':
        task.yields.append(['results', ev.get('images', [])])
    elif etype == 'error':
        task.processing = False
        task.yields.append(['error', ev.get('message') or 'Generation failed'])
        _cleanup_task(task)
        ipc.trim_memory()
    elif etype == 'finish':
        task.should_enhance = ev.get('should_enhance', False)
        task.enhance_stats = ev.get('enhance_stats', {})
        task.images_to_enhance_count = ev.get('images_to_enhance_count', 0)
        task.processing = False
        task.yields.append(['finish', ev.get('results', [])])
        _cleanup_task(task)
        ipc.trim_memory()
    elif etype == 'cancelled':
        task.processing = False
        task.yields.append(['error', 'Task cancelled'])
        _cleanup_task(task)
        ipc.trim_memory()


def _on_message(msg):
    tid = msg.get('id')
    if not tid:
        return
    with _lock:
        task = _tasks.get(tid)
    if task is None:
        return
    try:
        _dispatch(task, msg)
    except Exception as e:
        print(f'[remote_worker] bad event {msg.get("type")}: {e}', flush=True)


def _submit(task):
    if task.args is None or len(task.args) == 0:
        return
    task._id = uuid.uuid4().hex
    task.processing = True
    with _lock:
        _tasks[task._id] = task
    try:
        ipc.send({'cmd': 'submit', 'source': 'webui', 'id': task._id,
                  'payload': {'args': ipc._serialize_args(task.args)}})
    except Exception as e:
        task.processing = False
        task.yields.append(['error', f'Failed to submit task: {e}'])
        _cleanup_task(task)


class _TaskList(list):
    def append(self, task):
        list.append(self, task)
        _submit(task)


async_tasks = _TaskList()


def has_active_tasks():
    with _lock:
        return bool(_tasks)


def request_interrupt(value):
    """Ask the queue manager to stop/skip every queued and running WebUI task.
    The manager decides what to abort, so the buttons never kill the worker."""
    if value not in ('stop', 'skip'):
        return
    ipc.send({'cmd': value, 'source': 'webui'})


def kill_worker():
    """Ask the queue manager to terminate the generation worker now."""
    ipc.send({'cmd': 'kill'})


ipc.start_reader(_on_message)

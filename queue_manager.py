"""Queue manager: owns the generation queue, the single generation worker and
the two server clients (webui/api).

Everything lives in this process's memory:
  * the FIFO task queue (with its source webui|api),
  * the worker process (spawned on demand, terminated on idle/kill),
  * routing of worker events to the owning client.

No files, no ports: each child is a subprocess talking JSON-lines over its
stdin/stdout (see modules/ipc.py).
"""

import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
if ROOT not in sys.path:
    sys.path.append(ROOT)

SERVERS = os.environ.get('FOOOSTI_SERVERS', 'both')
API_PORT = int(os.environ.get('FOOOSTI_API_PORT', '8890') or '8890')
WEBUI_PORT = int(os.environ.get('FOOOSTI_WEBUI_PORT', '7865') or '7865')
LISTEN = os.environ.get('FOOOSTI_LISTEN', '127.0.0.1')
KEEPALIVE = float(os.environ.get('FOOOSTI_KEEPALIVE_MINUTES', '0') or '0')
REST = sys.argv[1:]

SOURCES = ('webui', 'api')
CLIENT_SPAWN_MIN_INTERVAL = 20.0
CLIENT_MAX_FAILS = 3

shutdown = threading.Event()


class Child:
    def __init__(self, name, proc):
        self.name = name
        self.proc = proc
        self.stdin = proc.stdin
        self.lock = threading.Lock()

    def send(self, msg):
        try:
            with self.lock:
                self.stdin.write((json.dumps(msg) + '\n').encode('utf-8'))
                self.stdin.flush()
        except Exception:
            pass


clients = {s: None for s in SOURCES}
worker = None
worker_ready = False
worker_busy = False
queue = deque()
current = None
owners = {}
idle_timer = None
state_lock = threading.RLock()
client_born = {}
client_fails = {}


def _read_lines(stream):
    while True:
        line = stream.readline()
        if not line:
            return
        line = line.decode('utf-8', 'replace').strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except Exception:
            continue


def _cancel_idle_timer():
    global idle_timer
    with state_lock:
        if idle_timer is not None:
            idle_timer.cancel()
            idle_timer = None


def _reap(proc):
    """Reap a terminated worker so it cannot linger as a zombie."""
    if proc is None:
        return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except Exception:
            pass
    except Exception:
        pass


def _terminate_worker():
    global worker, worker_ready, worker_busy
    with state_lock:
        if worker is None:
            return
        proc = worker.proc
        _fail_current('Worker stopped or crashed')
        worker = None
        worker_ready = False
        worker_busy = False
    try:
        proc.terminate()
    except Exception:
        pass
    _reap(proc)


def _start_worker():
    global worker, worker_ready, worker_busy
    with state_lock:
        cmd = [sys.executable, os.path.join(ROOT, 'generation_worker.py')] + REST
        proc = subprocess.Popen(cmd, cwd=ROOT, env=dict(os.environ),
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None)
        worker = Child('worker', proc)
        worker_ready = False
        worker_busy = False
    print(f'[queue_manager] started worker pid={proc.pid}', flush=True)
    threading.Thread(target=_worker_reader, args=(worker,), daemon=True).start()


def _fail_current(reason):
    global current
    with state_lock:
        if current is None:
            return
        tid = current['id']
        src = owners.pop(tid, None)
        client = clients.get(src)
        if client is not None:
            client.send({'type': 'error', 'id': tid, 'message': reason, 'results': []})
        current = None


def _ensure_worker():
    with state_lock:
        if worker is not None and worker.proc.poll() is None:
            return
        if worker is not None:
            # dead worker reference: fail its in-flight task before respawn
            _fail_current('Worker stopped or crashed')
            worker_ready = False
            worker_busy = False
        _start_worker()


def _assign():
    global current, worker_busy
    with state_lock:
        if current is not None or worker_busy or not queue:
            return
        _ensure_worker()
        if not worker_ready:
            return
        task = queue.popleft()
        current = {'id': task['id'], 'source': task['source'], 'worker': worker}
        owners[task['id']] = task['source']
        worker_busy = True
        _cancel_idle_timer()
        worker.send({'cmd': 'task', 'id': task['id'], 'source': task['source'], 'payload': task['payload']})


def _idle_timeout():
    global idle_timer
    with state_lock:
        if current is None and not queue and worker is not None \
                and worker.proc.poll() is None and worker_ready and not worker_busy:
            print('[queue_manager] idle timeout, stopping worker', flush=True)
            _terminate_worker()
        idle_timer = None


def _assign_or_idle():
    global idle_timer
    with state_lock:
        _assign()
        if current is not None or queue:
            _cancel_idle_timer()
            return
        _cancel_idle_timer()
        if KEEPALIVE <= 0:
            if worker is not None and worker.proc.poll() is None:
                print('[queue_manager] queue empty, stopping worker', flush=True)
                _terminate_worker()
        else:
            idle_timer = threading.Timer(KEEPALIVE * 60, _idle_timeout)
            idle_timer.daemon = True
            idle_timer.start()


def _on_worker_message(msg):
    global worker_ready, worker_busy, current
    mtype = msg.get('type')
    if mtype == 'ready':
        with state_lock:
            worker_ready = True
            _assign()
    elif mtype == 'task_done':
        with state_lock:
            tid = msg.get('id')
            if current is not None and current['id'] == tid:
                owners.pop(tid, None)
                current = None
            worker_busy = False
            _assign_or_idle()
    elif mtype == 'event':
        tid = msg.get('id')
        src = owners.get(tid)
        client = clients.get(src)
        if client is not None:
            client.send(msg.get('event', {}))


def _on_worker_eof(child):
    global worker, worker_ready, worker_busy
    with state_lock:
        if current is not None and current.get('worker') is child:
            _fail_current('Worker stopped or crashed')
        if worker is child:
            worker_ready = False
            worker_busy = False
            worker = None
        _assign()


def _control(source, action):
    with state_lock:
        removed = []
        keep = deque()
        for t in queue:
            if (action == 'stop' and t['source'] == source) or \
                    (action == 'skip' and t['source'] == source and not removed):
                removed.append(t)
            else:
                keep.append(t)
        queue.clear()
        queue.extend(keep)
        for t in removed:
            owners.pop(t['id'], None)
            client = clients.get(t['source'])
            if client is not None:
                client.send({'type': 'cancelled', 'id': t['id']})
        if current is not None and current['source'] == source:
            if worker is not None:
                worker.send({'cmd': 'control', 'action': action})
        _assign()


def _on_client_message(client, msg):
    cmd = msg.get('cmd')
    if cmd == 'submit':
        payload = msg.get('payload')
        if payload is None:
            return
        with state_lock:
            tid = msg.get('id') or uuid.uuid4().hex
            owners[tid] = client.name
            queue.append({'id': tid, 'source': client.name, 'payload': payload})
            _cancel_idle_timer()
            _assign()
    elif cmd in ('stop', 'skip'):
        _control(client.name, cmd)
    elif cmd == 'kill':
        print(f'[queue_manager] kill requested by {client.name}', flush=True)
        _terminate_worker()


def _on_client_eof(client):
    if clients.get(client.name) is client:
        clients[client.name] = None
    if shutdown.is_set():
        return
    print(f'[queue_manager] {client.name} client disconnected', flush=True)
    _control(client.name, 'stop')
    born = client_born.get(client.name, 0)
    if born and time.time() - born < CLIENT_SPAWN_MIN_INTERVAL:
        client_fails[client.name] = client_fails.get(client.name, 0) + 1
    if client_fails.get(client.name, 0) >= CLIENT_MAX_FAILS:
        print(f'[queue_manager] {client.name} keeps failing, not restarting', flush=True)
        return
    time.sleep(2)
    _start_client(client.name)


def _client_reader(client):
    for msg in _read_lines(client.proc.stdout):
        _on_client_message(client, msg)
    _on_client_eof(client)


def _worker_reader(child):
    for msg in _read_lines(child.proc.stdout):
        _on_worker_message(msg)
    _reap(child.proc)
    _on_worker_eof(child)


def _start_client(name):
    if name == 'api':
        cmd = [sys.executable, os.path.join(ROOT, 'launch_api.py'),
               '--port', str(API_PORT), '--listen', LISTEN] + REST
    else:
        cmd = [sys.executable, os.path.join(ROOT, 'launch_webui.py'),
               '--port', str(WEBUI_PORT), '--listen', LISTEN] + REST
    proc = subprocess.Popen(cmd, cwd=ROOT, env=dict(os.environ),
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None)
    client = Child(name, proc)
    clients[name] = client
    client_born[name] = time.time()
    print(f'[queue_manager] started {name} server pid={proc.pid}', flush=True)
    threading.Thread(target=_client_reader, args=(client,), daemon=True).start()


def _shutdown_all():
    _cancel_idle_timer()
    _terminate_worker()
    for name in list(clients):
        client = clients[name]
        if client is not None and client.proc.poll() is None:
            try:
                client.proc.terminate()
            except Exception:
                pass
    for name in list(clients):
        client = clients[name]
        if client is not None:
            try:
                client.proc.wait(timeout=10)
            except Exception:
                try:
                    client.proc.kill()
                except Exception:
                    pass


def _handle_signal(signum, frame):
    print(f'[queue_manager] got signal {signum}, shutting down', flush=True)
    shutdown.set()


def main():
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    if SERVERS in ('both', 'webui'):
        _start_client('webui')
    if SERVERS in ('both', 'api'):
        _start_client('api')
    print(f'[queue_manager] running (servers={SERVERS}, keepalive={KEEPALIVE}min)', flush=True)
    while not shutdown.is_set():
        time.sleep(1)
    _shutdown_all()


if __name__ == '__main__':
    main()

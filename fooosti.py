"""Fooosti daemon: owns the generation queue, the single generation worker and
the two server clients (webui/api).

Everything lives in this process's memory:
  * the FIFO task queue (with its source webui|api),
  * the worker process (spawned on demand, terminated on idle/kill),
  * routing of worker events to the owning client.

Clients are either always-alive or lazy (see WEBUI_KEEPALIVE_MINUTES /
API_KEEPALIVE_MINUTES below). A lazy client is not started at boot: the daemon
holds its port, spawns the client on the first incoming request (proxying that
first connection), and takes the port back when the client exits after idling.

No files, no extra ports: children talk JSON-lines over their stdin/stdout
(see modules/ipc.py); the only ports owned by the daemon are the lazy ones.
"""

import argparse
import json
import os
import signal
import socket
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
lazy_servers = {}
LAZY = set()


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
    print(f'[fooosti] started worker pid={proc.pid}', flush=True)
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
            print('[fooosti] idle timeout, stopping worker', flush=True)
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
        if WORKER_KEEPALIVE == -1:
            return
        if WORKER_KEEPALIVE <= 0:
            if worker is not None and worker.proc.poll() is None:
                print('[fooosti] queue empty, stopping worker', flush=True)
                _terminate_worker()
        else:
            idle_timer = threading.Timer(WORKER_KEEPALIVE * 60, _idle_timeout)
            idle_timer.daemon = True
            idle_timer.start()


def _on_worker_message(msg):
    global worker_ready, worker_busy, current
    mtype = msg.get('type')
    if mtype == 'ready':
        with state_lock:
            worker_ready = True
            # if nothing is queued anymore (task was cancelled while the worker
            # was still starting), terminate it instead of leaving a zombie
            _assign_or_idle()
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
        print(f'[fooosti] kill requested by {client.name}', flush=True)
        _terminate_worker()


def _on_client_eof(client):
    if clients.get(client.name) is client:
        clients[client.name] = None
    if shutdown.is_set():
        return
    print(f'[fooosti] {client.name} client disconnected', flush=True)
    _control(client.name, 'stop')
    if client.name in LAZY:
        lazy_servers[client.name].rebind()
        return
    born = client_born.get(client.name, 0)
    if born and time.time() - born < CLIENT_SPAWN_MIN_INTERVAL:
        client_fails[client.name] = client_fails.get(client.name, 0) + 1
    if client_fails.get(client.name, 0) >= CLIENT_MAX_FAILS:
        print(f'[fooosti] {client.name} keeps failing, not restarting', flush=True)
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
    print(f'[fooosti] started {name} server pid={proc.pid}', flush=True)
    threading.Thread(target=_client_reader, args=(client,), daemon=True).start()


def _pump(src, dst):
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        try:
            dst.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        try:
            src.close()
        except OSError:
            pass


class LazyServer:
    """Holds a port for a lazy client. The first incoming connection wakes it up:
    the port is released, the client is spawned on it and that one connection is
    proxied to it; afterwards the client owns the port until it exits (idle),
    at which point rebind() takes the port back."""

    def __init__(self, name, port):
        self.name = name
        self.port = port
        self.listener = None
        self.lock = threading.Lock()
        self.bound = False

    def start(self):
        self._bind()

    def _bind(self):
        with self.lock:
            if self.bound:
                return
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((LISTEN, self.port))
            s.listen(64)
            s.settimeout(1.0)
            self.listener = s
            self.bound = True
        print(f'[fooosti] lazy {self.name}: holding port {self.port}', flush=True)
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def _accept_loop(self):
        while not shutdown.is_set():
            s = self.listener
            if s is None:
                return
            try:
                conn, _ = s.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            self._on_first_conn(conn)

    def _peek(self, conn, timeout=0.5):
        conn.settimeout(timeout)
        chunks = []
        try:
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                chunks.append(data)
                if b'\r\n\r\n' in b''.join(chunks) or sum(len(c) for c in chunks) >= 65536:
                    break
        except socket.timeout:
            pass
        except OSError:
            pass
        finally:
            conn.settimeout(None)
        return b''.join(chunks)

    @staticmethod
    def _is_probe(data):
        """A health-check probe is answered by the daemon itself (200, no
        client spawn) so panels don't wake the lazy client on every poll.
        Real page loads: HEAD (never real for us), GET with Sec-Fetch-Dest:
        document or Accept: text/html, or an Upgrade (gradio websocket).
        JS fetches (mode: no-cors) send Sec-Fetch-Dest: empty + Accept: */*
        and are probes. POST and other methods are always real requests."""
        if not data:
            return True
        head = data.split(b'\r\n', 1)[0].decode('latin-1', 'replace')
        parts = head.split(' ')
        method = parts[0].upper() if parts else ''
        if method == 'HEAD':
            return True
        if method != 'GET':
            return False
        headers = data.decode('latin-1', 'replace').lower()
        if '\r\nupgrade:' in headers:
            return False
        if 'sec-fetch-dest: document' in headers:
            return False
        if 'accept: text/html' in headers:
            return False
        return True

    def _reply_probe(self, conn):
        if self.name == 'api':
            body = b'{"status":"ok"}'
            ctype = b'application/json'
        else:
            body = b'Fooosti webui is up (lazy, not loaded)'
            ctype = b'text/plain'
        resp = (b'HTTP/1.1 200 OK\r\n'
                b'Content-Type: ' + ctype + b'\r\n'
                b'Content-Length: ' + str(len(body)).encode() + b'\r\n'
                b'Connection: close\r\n'
                b'\r\n' + body)
        try:
            conn.sendall(resp)
        except OSError:
            pass
        try:
            conn.close()
        except OSError:
            pass

    def _on_first_conn(self, conn):
        data = self._peek(conn)
        if self._is_probe(data):
            print(f'[fooosti] lazy {self.name}: probe on {self.port}, answering 200 (no spawn)', flush=True)
            self._reply_probe(conn)
            return
        with self.lock:
            s, self.listener = self.listener, None
            self.bound = False
        try:
            s.close()
        except OSError:
            pass
        print(f'[fooosti] lazy {self.name}: request on {self.port}, spawning client', flush=True)
        _start_client(self.name)
        if not self._wait_ready(60):
            try:
                conn.close()
            except OSError:
                pass
            self._bind()
            return
        threading.Thread(target=self._proxy_first, args=(conn, data), daemon=True).start()

    def _wait_ready(self, timeout):
        deadline = time.time() + timeout
        while time.time() < deadline and not shutdown.is_set():
            client = clients.get(self.name)
            if client is None or client.proc.poll() is not None:
                time.sleep(0.1)
                continue
            try:
                s = socket.create_connection(('127.0.0.1', self.port), timeout=0.5)
                s.close()
                return True
            except OSError:
                time.sleep(0.1)
        return False

    def _proxy_first(self, conn, head=b''):
        try:
            up = socket.create_connection(('127.0.0.1', self.port), timeout=5)
        except OSError:
            try:
                conn.close()
            except OSError:
                pass
            return
        up.settimeout(None)
        if head:
            try:
                up.sendall(head)
            except OSError:
                for s in (conn, up):
                    try:
                        s.close()
                    except OSError:
                        pass
                return
        t1 = threading.Thread(target=_pump, args=(conn, up))
        t2 = threading.Thread(target=_pump, args=(up, conn))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        for s in (conn, up):
            try:
                s.close()
            except OSError:
                pass

    def rebind(self):
        self._bind()

    def close(self):
        with self.lock:
            s, self.listener = self.listener, None
            self.bound = False
        if s is not None:
            try:
                s.close()
            except OSError:
                pass


def _shutdown_all():
    _cancel_idle_timer()
    _terminate_worker()
    for name in list(lazy_servers):
        lazy_servers[name].close()
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
    print(f'[fooosti] got signal {signum}, shutting down', flush=True)
    shutdown.set()


def _keepalive(name, default):
    try:
        raw = os.environ.get(name)
        return float(raw) if raw else default
    except ValueError:
        return default


def parse_args(argv):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--listen', type=str, default='127.0.0.1', nargs='?', const='0.0.0.0')
    p.add_argument('--api-port', type=int, default=8890)
    p.add_argument('--webui-port', type=int, default=7865)
    p.add_argument('--only-api', action='store_true')
    p.add_argument('--only-webui', action='store_true')
    ns, rest = p.parse_known_args(argv)
    return ns, rest


def main():
    global SERVERS, API_PORT, WEBUI_PORT, LISTEN, REST, WORKER_KEEPALIVE, WEBUI_KEEPALIVE, API_KEEPALIVE, LAZY

    ns, rest = parse_args(sys.argv[1:])
    REST = rest
    API_PORT = ns.api_port
    WEBUI_PORT = ns.webui_port
    LISTEN = ns.listen

    if ns.only_api:
        SERVERS = 'api'
    elif ns.only_webui:
        SERVERS = 'webui'
    elif os.environ.get('FOOOSTI_WEBUI', '1') == '0':
        SERVERS = 'api'
    else:
        SERVERS = 'both'

    WORKER_KEEPALIVE = _keepalive('WORKER_KEEPALIVE_MINUTES', 0.0)
    WEBUI_KEEPALIVE = _keepalive('WEBUI_KEEPALIVE_MINUTES', -1.0)
    API_KEEPALIVE = _keepalive('API_KEEPALIVE_MINUTES', -1.0)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    if SERVERS in ('both', 'webui'):
        if WEBUI_KEEPALIVE == -1:
            _start_client('webui')
        else:
            LAZY.add('webui')
            lazy_servers['webui'] = LazyServer('webui', WEBUI_PORT)
            lazy_servers['webui'].start()
    if SERVERS in ('both', 'api'):
        if API_KEEPALIVE == -1:
            _start_client('api')
        else:
            LAZY.add('api')
            lazy_servers['api'] = LazyServer('api', API_PORT)
            lazy_servers['api'].start()

    print(f'[fooosti] running (servers={SERVERS}, lazy={sorted(LAZY) or "none"}, '
          f'keepalive worker={WORKER_KEEPALIVE}min, webui={WEBUI_KEEPALIVE}min, api={API_KEEPALIVE}min)',
          flush=True)

    while not shutdown.is_set():
        time.sleep(1)
    _shutdown_all()


if __name__ == '__main__':
    main()

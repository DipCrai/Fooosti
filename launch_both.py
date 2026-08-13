import argparse
import os
import signal
import subprocess
import sys
import time

from modules import constants

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)


def parse_args(argv):
    # --port applies to the API; --listen defaults to loopback for security.
    # Everything else is forwarded verbatim to launch_api.py / launch.py (and
    # from there to the generation workers), so --gpu-device-id, --preset,
    # --disable-image-log, --share, ... all keep working.
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--listen', type=str, default='127.0.0.1', nargs='?', const='0.0.0.0')
    p.add_argument('--port', type=int, default=8890)
    ns, rest = p.parse_known_args(argv)
    return ns, rest


def main():
    ns, rest = parse_args(sys.argv[1:])
    api_port = ns.port
    listen = ns.listen
    webui_port = 7865
    start_webui = os.environ.get('FOOOSTI_WEBUI', '1') != '0'

    procs = {}

    def start(name, cmd, env):
        print(f'[launcher] starting {name}: {" ".join(cmd)}', flush=True)
        procs[name] = subprocess.Popen(cmd, cwd=ROOT, env=env)
        return procs[name]

    # isolate each server's worker control dir so UI/API stop files never clash
    tmp_base = os.environ.get('FOOOSTI_TMP_DIR', constants.FOOOSTI_TMP_DIR)
    api_env = {**os.environ, 'FOOOSTI_TMP_DIR': os.path.join(tmp_base, 'api')}
    webui_env = {**os.environ, 'FOOOSTI_TMP_DIR': os.path.join(tmp_base, 'ui')}
    os.makedirs(api_env['FOOOSTI_TMP_DIR'], exist_ok=True)
    os.makedirs(webui_env['FOOOSTI_TMP_DIR'], exist_ok=True)

    api = start('api', [sys.executable, os.path.join(ROOT, 'launch_api.py'),
                        '--port', str(api_port), '--listen', listen, *rest], api_env)
    if start_webui:
        start('webui', [sys.executable, os.path.join(ROOT, 'launch.py'),
                        '--port', str(webui_port), '--listen', listen, *rest], webui_env)

    stop = False

    def _term(signum, frame):
        nonlocal stop
        stop = True
        for name, p in list(procs.items()):
            if p.poll() is None:
                print(f'[launcher] stopping {name}', flush=True)
                p.terminate()

    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)

    try:
        while not stop:
            for name, p in list(procs.items()):
                rc = p.poll()
                if rc is not None:
                    print(f'[launcher] {name} exited rc={rc}', flush=True)
                    stop = True
                    break
            if stop:
                break
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for name, p in list(procs.items()):
            if p.poll() is None:
                try:
                    p.terminate()
                    p.wait(timeout=10)
                except Exception:
                    p.kill()
        for p in procs.values():
            p.wait(timeout=10)


if __name__ == '__main__':
    main()

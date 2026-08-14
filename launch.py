"""Single entry point / supervisor for Fooosti.

Starts the queue manager (which owns the generation worker and the webui/api
servers) and relays termination signals. Use --only-api/--only-webui to run a
single server; everything else is passed through to the children.
"""

import argparse
import os
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)


def parse_args(argv):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--listen', type=str, default='127.0.0.1', nargs='?', const='0.0.0.0')
    p.add_argument('--api-port', type=int, default=8890, help='API port (default 8890)')
    p.add_argument('--webui-port', type=int, default=7865, help='WebUI port (default 7865)')
    p.add_argument('--only-api', action='store_true', help='run only the API server')
    p.add_argument('--only-webui', action='store_true', help='run only the WebUI server')
    ns, rest = p.parse_known_args(argv)
    return ns, rest


def main():
    ns, rest = parse_args(sys.argv[1:])

    if ns.only_api:
        servers = 'api'
    elif ns.only_webui:
        servers = 'webui'
    elif os.environ.get('FOOOSTI_WEBUI', '1') == '0':
        servers = 'api'
    else:
        servers = 'both'

    env = {
        **os.environ,
        'FOOOSTI_SERVERS': servers,
        'FOOOSTI_API_PORT': str(ns.api_port),
        'FOOOSTI_WEBUI_PORT': str(ns.webui_port),
        'FOOOSTI_LISTEN': ns.listen,
    }

    cmd = [sys.executable, os.path.join(ROOT, 'queue_manager.py')] + rest
    print(f'[launch] starting queue manager: {" ".join(cmd)}', flush=True)
    manager = subprocess.Popen(cmd, cwd=ROOT, env=env)

    def _term(signum, frame):
        if manager.poll() is None:
            print('[launch] relaying signal to queue manager', flush=True)
            manager.terminate()

    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)

    try:
        rc = manager.wait()
    except KeyboardInterrupt:
        rc = None
    finally:
        if manager.poll() is None:
            try:
                manager.terminate()
                manager.wait(timeout=10)
            except Exception:
                manager.kill()

    if rc is None:
        rc = 0
    print(f'[launch] queue manager exited rc={rc}', flush=True)
    sys.exit(rc)


if __name__ == '__main__':
    main()

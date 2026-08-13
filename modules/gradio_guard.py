# ---- Fooosti: disable gradio's built-in JSON API (use the dedicated /sdapi/v1 API instead) ----
#
# routes (incl. /run/{api_name} and /api/{api_name}) are registered in
# App.create_app AFTER App.configure_app, so the filter must wrap create_app.
# /run/{api_name} and the /queue/join websocket are the gradio frontend's own
# protocol (non-queue + queue events) and must stay for the UI to work at all.
# Only the /api/ REST surface is removed so external API clients cannot drive
# the UI. The gradio queue worker itself calls POST /api/predict internally
# (queueing.Queue.call_prediction), so that single path is kept reachable and
# hidden from openapi; its request-time queue-token check still applies. Bind
# to loopback or enable gradio auth (modules/auth.py) to lock /run/ too.

import inspect
import os
import urllib.parse

import gradio.routes as _gr_routes
import starlette.responses as _starlette_responses

_SENSITIVE_SUFFIXES = ('.py', '.sh', '.bat', '.csh', '.safetensors', '.ckpt', '.pth', '.pt', '.bin')
_SENSITIVE_BASENAMES = {'config.txt', 'Dockerfile', '.env', 'auth.json',
                        'auth-example.json', 'config_modification_tutorial.txt',
                        'docker-compose.yml', 'user_path_config.txt'}


def _blocked_paths(app_root):
    data_dir = os.environ.get('DATADIR', '/content/data')
    candidates = [
        os.environ.get('config_path', os.path.join(data_dir, 'config.txt')),
        os.path.join(data_dir, 'config.txt'),
        os.path.join(app_root, 'config.txt'),
        os.path.join(data_dir, 'models'),
        os.path.join(app_root, 'presets'),
        os.path.join(app_root, 'models'),
    ]
    # dedupe while preserving order: config.txt can legitimately resolve to the
    # same path twice (config_path env set to the data_dir default)
    return list(dict.fromkeys(candidates))


def install(app_root):
    create_app_original = _gr_routes.App.create_app

    def fooosti_create_app(blocks, app_kwargs=None):
        app = create_app_original(blocks, app_kwargs)

        pred_endpoint = None
        pred_deps = None
        kept_routes = []
        for route in app.routes:
            if getattr(route, 'path', None) in ('/api/{api_name}', '/api/{api_name}/'):
                if pred_endpoint is None:
                    pred_endpoint = route.endpoint
                    pred_deps = list(getattr(route, 'dependencies', [])) or None
                continue
            kept_routes.append(route)
        app.routes[:] = kept_routes
        if pred_endpoint is not None:
            # the queue worker calls POST /api/predict internally
            # (queueing.Queue.call_prediction); gradio's signature can change
            # across versions, so fall back to calling without 'username'
            try:
                params = inspect.signature(pred_endpoint).parameters
                username_default = params['username'].default if 'username' in params else None
            except Exception:
                username_default = None

            if username_default is not None:
                async def fooosti_api_predict(body: _gr_routes.PredictBody,
                                              request: _gr_routes.fastapi.Request,
                                              username=username_default):
                    return await pred_endpoint(api_name='predict', body=body,
                                               request=request, username=username)
            else:
                async def fooosti_api_predict(body: _gr_routes.PredictBody,
                                              request: _gr_routes.fastapi.Request):
                    return await pred_endpoint(api_name='predict', body=body, request=request)

            app.add_api_route('/api/predict', fooosti_api_predict, methods=['POST'],
                              dependencies=pred_deps, include_in_schema=False)
        else:
            print('[Fooosti] WARNING: gradio /api/{api_name} route not found - '
                  'gradio version changed, /api/predict was NOT re-registered. '
                  'The UI queue may be broken.', flush=True)

        # never serve sensitive files/dirs via /file=
        blocks.blocked_paths = list(blocks.blocked_paths or []) + _blocked_paths(app_root)

        @app.middleware('http')
        async def fooosti_block_sensitive_requests(request, call_next):
            if request.url.path == '/' and request.query_params.get('__view') == 'api':
                return _starlette_responses.Response(status_code=404)
            if request.url.path.startswith('/file='):
                fp = request.url.path[len('/file='):].split('?')[0]
                # fully decode so double-encoded traversal (%252e) cannot slip past
                while True:
                    dec = urllib.parse.unquote(fp)
                    if dec == fp:
                        break
                    fp = dec
                base = os.path.basename(fp)
                if base in _SENSITIVE_BASENAMES or os.path.splitext(base)[1].lower() in _SENSITIVE_SUFFIXES:
                    return _starlette_responses.Response(status_code=403)
            response = await call_next(request)
            if request.url.path.startswith('/file=') and request.url.path.lower().endswith('.html'):
                response.headers['X-Content-Type-Options'] = 'nosniff'
            return response

        return app

    _gr_routes.App.create_app = staticmethod(fooosti_create_app)

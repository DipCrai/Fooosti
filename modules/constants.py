# as in k-diffusion (sampling.py)
MIN_SEED = 0
MAX_SEED = 2**63 - 1

AUTH_FILENAME = 'auth.json'

# maximum wall-clock time a single generation may run before it is aborted
FOOOSTI_GENERATION_TIMEOUT = 7200

# directory shared between the API/webui launcher, the worker and the UI
FOOOSTI_TMP_DIR = '/tmp/fooosti_api'

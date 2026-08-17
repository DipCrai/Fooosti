# Fooosti on Docker

The docker image inherits CUDA 12.4 and PyTorch 2.1 from the upstream image `ghcr.io/lllyasviel/fooocus:edge` it is built on; see [Dockerfile](Dockerfile) and [requirements_docker.txt](requirements_docker.txt) for details.

This is a fork of Fooocus tuned to run as a background service: the image runs `fooosti.py`, which starts the WebUI (7865), the API (8890) and a shared generation worker through a single daemon. The image is built locally from this repository (no published `fooosti` image); the build pulls the upstream `fooocus:edge` base, see [Building the container locally](#building-the-container-locally).

## Requirements

- A computer with specs good enough to run Fooocus, and proprietary Nvidia drivers
- Docker, Docker Compose, or Podman

## Quick start

**More information in the [notes](#notes).**

### Running with Docker Compose (recommended)

1. Clone this repository
2. Run `docker compose up -d --build`.

Web UI: http://localhost:7865 · API: http://localhost:8890 (docs at `/docs`) · data in the `fooocus-data` volume.

Set `CMDARGS=--serves webui` to run only the WebUI, or `CMDARGS=--serves api` for only the API. The default `both` runs both.

### Running with Docker

The Docker and Podman commands below are identical except for the runtime flags.

```sh
docker run -p 7865:7865 -p 8890:8890 -v fooocus-data:/content/data -it \
--gpus all \
-e CMDARGS=--listen \
-e DATADIR=/content/data \
-e config_path=/content/data/config.txt \
-e config_example_path=/content/data/config_modification_tutorial.txt \
-e path_checkpoints=/content/data/models/checkpoints/ \
-e path_loras=/content/data/models/loras/ \
-e path_embeddings=/content/data/models/embeddings/ \
-e path_vae_approx=/content/data/models/vae_approx/ \
-e path_upscale_models=/content/data/models/upscale_models/ \
-e path_inpaint=/content/data/models/inpaint/ \
-e path_controlnet=/content/data/models/controlnet/ \
-e path_clip_vision=/content/data/models/clip_vision/ \
-e path_fooocus_expansion=/content/data/models/prompt_expansion/fooocus_expansion/ \
-e path_outputs=/content/app/outputs/ \
fooosti:latest
```

### Running with Podman

Podman uses the same flags as the `docker run` command above, with the runtime flags replaced:

```sh
podman run -p 7865:7865 -p 8890:8890 -v fooocus-data:/content/data -it \
--security-opt=no-new-privileges --cap-drop=ALL \
--security-opt label=type:nvidia_container_t --device=nvidia.com/gpu=all \
fooosti:latest
```

(plus the same `-e` environment flags as the Docker command.)

When you see the message `Use the app with http://0.0.0.0:7865/` in the console, you can access the URL in your browser.

Your models and outputs are stored in the `fooocus-data` volume, which, depending on OS, is stored in `/var/lib/docker/volumes/` (or `~/.local/share/containers/storage/volumes/` when using `podman`).

## Building the container locally

Clone the repository first, and open a terminal in the folder.

Build with `docker`:
```sh
docker build . -t fooosti:latest
```

Build with `podman`:
```sh
podman build . -t fooosti:latest
```

## Details

### Update the container manually (`docker compose`)

When you are using `docker compose up` continuously, the container is not updated to the latest version of Fooocus automatically.
Run `git pull` before executing `docker compose build --no-cache` to build an image with the latest Fooocus version.
You can then start it with `docker compose up`

### Import models, outputs

If you want to import files from models or the outputs folder, you can add the following bind mounts in the [docker-compose.yml](docker-compose.yml) or your preferred method of running the container:
```
#- ./models:/import/models   # Once you import files, you don't need to mount again.
#- ./outputs:/import/outputs  # Once you import files, you don't need to mount again.
```
After running the container, your files will be copied into `/content/data/models` and `/content/data/outputs`
Since `/content/data` is a persistent volume folder, your files will be persisted even when you re-run the container without the above mounts.


### Paths inside the container

|Path|Details|
|-|-|
|/content/app|The application stored folder|
|/content/app/models.org|Original 'models' folder.<br> Files are copied to the '/content/app/models' which is symlinked to '/content/data/models' every time the container boots. (Existing files will not be overwritten.) |
|/content/data|Persistent volume mount point|
|/content/data/models|The folder is symlinked to '/content/app/models'|
|/content/data/outputs|The folder is symlinked to '/content/app/outputs'|

### Command line flags (CMDARGS)

Flags are passed via `CMDARGS` in `docker-compose.yml` and only reach the daemon — they configure the daemon itself and are not inherited by its subprocesses. Keep them for daemon-only, startup-time settings:

|Flag|Details|
|-|-|
|`--api-port`|API port (default 8890)|
|`--webui-port`|WebUI port (default 7865)|
|`--listen`|Bind address (default 127.0.0.1; `--listen` alone = `0.0.0.0`)|
|`--serves`|What the daemon serves: `both` (default), `webui`, `api`|

Example: `CMDARGS=--api-port 8890 --webui-port 7865 --listen 127.0.0.1 --serves both`

### Environment variables

Env vars are inherited by the daemon's subprocesses (worker, webui, api clients), so anything those processes must read lives here. They also cover build-time settings needed before Python starts (Dockerfile).

|Environment|Details|
|-|-|
|FOOOSTI_API_TOKEN|Optional API auth token (see [API.md](API.md#authentication))|
|WORKER_KEEPALIVE_MINUTES|How long the worker stays loaded after a task; `0` = unload after every task; `-1` = always alive|
|WEBUI_KEEPALIVE_MINUTES|WebUI client lifetime: `-1` = always alive (default), `0` = lazy (spawned on first visit, exits once the last browser tab closes and no generation is running), `N>0` = die after N minutes with no open tab|
|API_KEEPALIVE_MINUTES|API client lifetime: `-1` = always alive (default), `0` = lazy (spawned on first request, exits once the last response is fully sent), `N>0` = die after N minutes idle|
|IDLE_TICK|Idle-monitor poll interval in seconds (default 0.2), shared by webui and api — how quickly a closed tab / finished request is noticed|
|FOOOSTI_GENERATION_TIMEOUT|API generation timeout in seconds (default 7200)|
|DATADIR|'/content/data' location|
|HF_MIRROR| huggingface mirror site domain| 

You can also use the same json key names and values explained in the 'config_modification_tutorial.txt' as the environments.
See examples in the [docker-compose.yml](docker-compose.yml)

## Notes

- Please keep 'path_outputs' under '/content/app'. Otherwise, you may get an error when you open the history log.
- Docker on Mac/Windows still has issues in the form of slow volume access when you use "bind mount" volumes. Please refer to [this article](https://docs.docker.com/storage/volumes/#use-a-volume-with-docker-compose) for not using "bind mount".
- You can also use `docker compose up -d` to start the container detached and connect to the logs with `docker compose logs -f`. This way you can also close the terminal and keep the container running.
# Fooosti - Fast Fooocus API

Fooosti is a self-hosted fork of [Fooocus](https://github.com/lllyasviel/Fooocus), tuned to run as a lightweight background service in Docker. It keeps the familiar Fooocus image quality and UI, and adds a clean HTTP API on top.

## Why a fork?

- **One process owns everything.** A single `fooosti` daemon launches the WebUI, the API and the generation worker (a subprocess) — no fragile chain of scripts to keep alive.
- **Runs idle for almost nothing.** Generation happens in a worker subprocess that is spawned per task and torn down when done, so the main process sits nearly idle in between — no tens of gigabytes pinned in VRAM while you do nothing else on the GPU. The worker lifetime is controlled by the `WORKER_KEEPALIVE_MINUTES` env var (`0` = unload the worker after every task).
- **Lazy clients by default.** The WebUI and API can live inside the daemon (always-alive, ~176 MiB idle) or be spawned on demand (`WEBUI_KEEPALIVE_MINUTES` / `API_KEEPALIVE_MINUTES`, `0` = lazy): the daemon holds their port, starts the client on the first request and takes the port back after it exits on idle — leaving the container at 3-11 MiB (depending on swap) until something touches it.
- **A real API.** FastAPI server on port `8890` with A1111-compatible `/sdapi/v1/*` endpoints — so it plugs straight into [Open WebUI](https://github.com/open-webui/open-webui) (or any other A1111 client) as a drop-in backend. API requests are queued and processed one at a time by the single worker. See [API.md](API.md) for endpoints, request templates and auth.
- **Local prompt translation & enhancement.** Non-English prompts are rewritten into detailed English SD prompts by a local LLM — **Qwen2.5-1.5B-Instruct-abliterated**, downloaded from [HuggingFace](https://huggingface.co/huihui-ai/Qwen2.5-1.5B-Instruct-abliterated). Works on top of the stock Fooocus V2 expansion. Fully configurable via `config.txt`: `enable_prompt_translator` disables it entirely, `prompt_translator_enhance_english` also enhances English prompts.
- **Redesigned UI.** Custom CSS and JS on top of the Gradio interface, UI settings persisted across restarts.

## Screenshots

| Fullscreen | Medium screen | Mobile screen |
|------------|---------------|---------------|
| <img src="image-1.png" height="420"> | <img src="image-2.png" height="420"> | <img src="image-3.png" height="420"> |

## Quick start (Docker)

Requirements: Nvidia GPU with proprietary drivers + Docker with the NVIDIA Container Toolkit.

```sh
mkdir -p ~/Fooosti
cd ~/Fooosti
docker compose up -d --build
```

- Web UI (Gradio): http://localhost:7865
- API (FastAPI): http://localhost:8890 (interactive docs at `/docs`)
- Data lives on the persistent `fooocus-data` Docker volume (mounted at `/content/data`) — checkpoints, LoRAs, outputs, config and the translator model survive container rebuilds.

Drop `.safetensors` checkpoints into the volume with `docker compose cp model.safetensors app:/content/data/models/checkpoints/`, press **Refresh All Files** in the UI (Models tab, bottom of the right column), and you're done.

To run only one part of the stack, set `CMDARGS=--serves webui` or `CMDARGS=--serves api` in `docker-compose.yml` (`--serves both` is the default).

## Auth

The API can be locked with the `FOOOSTI_API_TOKEN` env var in `docker-compose.yml` (optional; empty by default). When set, every `/sdapi/v1/*` request must send the token in the `x-api-token` header. Empty token = unauthenticated. Details: [API.md](API.md#authentication).

## More docs

- [API.md](API.md) — endpoints, request templates, auth.
- [docker.md](docker.md) — running with plain Docker, Podman.
- [Upstream Fooocus README](https://github.com/lllyasviel/Fooocus/blob/main/readme.md) — original docs, install guides, FAQ and troubleshooting (swap, VRAM, CUDA errors).

## License

GPL-3.0 — inherited from the upstream [lllyasviel/Fooocus](https://github.com/lllyasviel/Fooocus). See [LICENSE](LICENSE).

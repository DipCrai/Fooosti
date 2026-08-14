# Fooosti API

A1111-compatible HTTP API served by FastAPI. Default port **8890**, interactive docs at `http://localhost:8890/docs`.

- Base URL: `http://<host>:8890`
- All `/sdapi/v1/*` endpoints are protected by the API token (see [Authentication](#authentication)).
- Generation is queued: the API shares the single worker with the WebUI through the queue manager. A request while another one is running waits in the FIFO queue instead of failing.
- The API is wired for the model selected via `POST /sdapi/v1/options` (used by Open WebUI) or the `default_model` key in `config.txt`.

## Authentication

Set `FOOOSTI_API_TOKEN` in `docker-compose.yml` to require a token.

Every `/sdapi/v1/*` request must then send it in the **`x-api-token`** header:

```
x-api-token: your-secret-token
```

- Wrong or missing token → `401` `{"detail": "invalid API token"}`.
- Empty `FOOOSTI_API_TOKEN` → endpoints are unauthenticated (the server prints a warning on startup).
- Comparison uses a constant-time check (`hmac.compare_digest`).

## Endpoints

### `GET /`

Landing page / banner. No auth.

Response — plain text:

```text
Fooosti is running
```

### `GET /health`

Health probe. No auth.

Response:

```json
{ "status": "ok" }
```

### `GET /sdapi/v1/options`

Current options (A1111-compatible).

Response:

```json
{
  "sd_model_checkpoint": "juggernautXL_v8Rundiffusion.safetensors",
  "sd_vae": "None"
}
```

### `POST /sdapi/v1/options`

Persist options. A1111 clients (e.g. Open WebUI) POST back the dict they got from GET to switch the model.

Request:

```json
{
  "sd_model_checkpoint": "NoobAI-XL-Vpred-v1.0.safetensors"
}
```

- Only `sd_model_checkpoint` is applied — it is written to `default_model` in `config.txt` and used by the next generation.
- Response: the updated options object (same shape as `GET /sdapi/v1/options`).
- Invalid body → `400`.

### `GET /sdapi/v1/sd-models`

List available checkpoints from `data/models/checkpoints/`.

Response:

```json
[
  { "title": "juggernautXL_v8Rundiffusion.safetensors", "model_name": "juggernautXL_v8Rundiffusion.safetensors", "hash": "" }
]
```

### `POST /sdapi/v1/txt2img`

Generate an image. Request body (`Txt2ImgRequest`):

Request:

```json
{
  "prompt": "a beautiful landscape at sunset, vibrant colors, detailed",
  "negative_prompt": "worst quality, low quality, blurry, watermark",
  "steps": 28,
  "width": 832,
  "height": 1216,
  "batch_size": 1,
  "cfg_scale": 4.5,
  "seed": -1,
  "sampler_name": "euler",
  "scheduler_name": "simple",
  "style_selections": ["Fooocus V2"],
  "performance": "Speed",
  "base_model_name": "NoobAI-XL-Vpred-v1.0.safetensors",
  "sharpness": 2.0,
  "metadata_scheme": "none"
}
```

| Field | Type | Constraints |
|---|---|---|
| `prompt` | string | max 10000, default `""` |
| `negative_prompt` | string | max 10000, default `""` |
| `steps` | int | 1–200 |
| `width` | int | 16–2048 |
| `height` | int | 16–2048 |
| `batch_size` | int | 1–8 |
| `cfg_scale` | float | 1.0–30.0 |
| `seed` | int | `-1` = random |
| `sampler_name` | string | any known sampler, e.g. `euler`, `dpmpp_2m_sde` |
| `scheduler_name` | string | any known scheduler, e.g. `simple`, `karras` |
| `style_selections` | string[] | style presets |
| `performance` | string | e.g. `Speed`, `Quality` |
| `base_model_name` | string | checkpoint filename only; path components are stripped |
| `sharpness` | float | |
| `metadata_scheme` | string | |

All fields optional — omitted ones use current defaults from `config.txt`.

Response:

```json
{
  "images": ["data:image/png;base64,iVBORw0KGgo..."],
  "parameters": { "prompt": "...", "...": "..." },
  "info": "{\"prompt\": \"...\"}"
}
```

- `images` — array of base64-encoded data URIs (one per batch).
- Errors: `500` with `{"detail": "..."}` on failure or timeout.
- The request waits in the queue while another generation is running (no `429`); the timeout includes queue wait time.
- Generation timeout default: 7200 s (`FOOOSTI_GENERATION_TIMEOUT`).

### `GET /sdapi/v1/progress`

Live progress of the in-flight generation.

Request:

```
GET /sdapi/v1/progress?skip_current_image=false
```

Response:

```json
{
  "progress": 0.42,
  "eta_relative": 12.5,
  "state": {
    "skipped": false,
    "interrupted": false,
    "job": "txt2img",
    "job_count": 1,
    "job_no": 0,
    "job_timestamp": "",
    "sampling_step": 12,
    "sampling_steps": 28
  },
  "current_image": null
}
```

- `current_image` is a base64 preview; pass `skip_current_image=true` to omit it.
- Idle (no task running) → `progress: 0.0`.

### `POST /sdapi/v1/interrupt`

Abort the in-flight generation. Request body: `{}`. Response: `{}`.

### `POST /sdapi/v1/skip`

Skip the current image of the in-flight generation. Request body: `{}`. Response: `{}`.

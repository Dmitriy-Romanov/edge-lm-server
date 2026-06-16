# edge-lm-server

OpenAI-compatible local gateway for running Edge-LM Gemma models with Pi Agent
on Apple silicon Macs.

The launcher prepares a local Python venv, starts the MLX-based Edge-LM server,
and exposes it at:

```bash
http://127.0.0.1:8000/v1
```

## Requirements

- macOS on Apple silicon
- Python 3.10 or newer

## Quick start

Clone the project and run the menu:

```bash
git clone https://github.com/Dmitriy-Romanov/edge-lm-server.git
cd edge-lm-server
./run
```

`./run` creates `.edge-lm-server/.venv`, installs Python dependencies there,
and opens a small menu. The menu asks:

- whether to start existing local model files, if they are present
- whether to show Pi Agent setup instructions
- whether to download/install a model into `models/`
- which model to run

Model files are downloaded from the upstream TheStageAI repositories on Hugging
Face. This repository intentionally does not distribute model weights through
GitHub LFS, because public LFS bandwidth can be exhausted and break installs.

The download/install menu option fetches the selected model into `models/`.
That directory is ignored by git and can be copied to another Mac. The Python
venv and package cache live under `.edge-lm-server/` and can be deleted
independently.

Advanced users can still run directly from a Hugging Face model id with
`--prefer-remote`, but the interactive menu intentionally installs models into
`models/` so downloaded files are visible and portable.

If you already ran this project before and want to refresh the upstream
`TheStageAI/edge-lm` Python package, run:

```bash
./run --reinstall
```

Python dependencies are intentionally not pinned. The local venv is disposable:
delete `.edge-lm-server/` or run `./run --reinstall` to get the latest available
upstream packages.

## Models

The menu currently supports the upstream QAT model ids:

- `TheStageAI/gemma-4-E4B-it-qat`
- `TheStageAI/gemma-4-E2B-it-qat`

These repositories also advertise GGUF files upstream, but this gateway uses
the native Edge-LM / MLX checkpoint path, not llama.cpp. GGUF support would be a
separate backend.

The install menu can download four local variants: E4B/E2B in `m` or `l` size.
The start menu only shows variants that are already installed under `models/`.

Approximate install sizes:

- E4B QAT `m`: about 3.1 GB
- E4B QAT `l`: about 3.7 GB
- E2B QAT `m`: about 1.8 GB
- E2B QAT `l`: about 2.1 GB

Some shared files are also needed, such as the tokenizer, audio tower, and
vision tower.

## Backend support

Current backend:

```text
MLX / Edge-LM safetensors
  -> Apple silicon macOS
```

Potential future backend, not implemented yet:

```text
GGUF / llama.cpp
  -> macOS, Linux, Windows
```

If GGUF support is added later, the menu could become:

```text
Choose backend:
  1) Edge-LM MLX (Apple silicon, QAT safetensors)
  2) llama.cpp GGUF (macOS/Linux/Windows)
```

## Pi Agent config

Add this provider to `~/.pi/agent/models.json`:

```json
{
  "local-edge": {
    "baseUrl": "http://127.0.0.1:8000/v1",
    "api": "openai-completions",
    "apiKey": "local-key",
    "compat": {
      "supportsDeveloperRole": false,
      "supportsReasoningEffort": false,
      "supportsUsageInStreaming": true
    },
    "models": [
      {
        "id": "TheStageAI/gemma-4-E4B-it-qat",
        "contextWindow": 128000,
        "maxTokens": 16000,
        "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 }
      },
      {
        "id": "TheStageAI/gemma-4-E2B-it-qat",
        "contextWindow": 128000,
        "maxTokens": 16000,
        "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 }
      }
    ]
  }
}
```

## Notes

The server currently uses Apple's MLX runtime through Edge-LM, so it is intended
for Apple silicon/macOS rather than Linux Docker or Windows.

See [NOTICE.md](NOTICE.md) for model attribution and terms.
Developer and maintainer notes live in [AGENTS.md](AGENTS.md).

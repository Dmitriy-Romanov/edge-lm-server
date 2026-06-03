# edge-lm-server

OpenAI-compatible local gateway for running Edge-LM Gemma models with Pi Agent
on Apple silicon Macs.

The launcher prepares a local Python runtime, starts the MLX-based Edge-LM
server, and exposes it at:

```bash
http://127.0.0.1:8000/v1
```

This repository includes vendored model files through Git LFS, so the default
model can still run even if the upstream model later disappears.

## Requirements

- macOS on Apple silicon
- Rust/Cargo
- Python 3.10 or newer
- Git LFS, only if you want to use the vendored model files

## Quick start

Use the vendored model from this repository:

```bash
git lfs install
git lfs pull
cargo build --release
./target/release/edge-lm-server
```

On first run, the launcher creates `.edge-lm-server`, installs the Python
dependencies, reassembles split model files if needed, and starts the gateway.

Or skip Git LFS and load the model from the remote source instead:

```bash
cargo build --release
./target/release/edge-lm-server --prefer-remote
```

This does not download the vendored model files from the repository, but it does
depend on the upstream model still being available.

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
        "id": "TheStageAI/gemma-4-E4B-it",
        "contextWindow": 128000,
        "maxTokens": 16000,
        "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 }
      }
    ]
  }
}
```

## Useful commands

```bash
./target/release/edge-lm-server --port 8001
./target/release/edge-lm-server --context 128000
./target/release/edge-lm-server --prefer-remote
./target/release/edge-lm-server clean
```

`--prefer-remote` ignores the vendored model and asks Edge-LM to load the model
from its remote source. This is useful when testing a newer upstream model.

## Notes

The server uses Apple's MLX runtime through Edge-LM, so it is intended for
Apple silicon/macOS rather than Linux Docker.

See [NOTICE.md](NOTICE.md) for model attribution and terms.
Developer and maintainer notes live in [AGENTS.md](AGENTS.md).

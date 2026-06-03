# edge-lm-server

OpenAI-compatible local gateway for running Edge-LM Gemma models with Pi Agent
on Apple silicon Macs.

The launcher prepares a local Python runtime, starts the MLX-based Edge-LM
server, and exposes it at:

```bash
http://127.0.0.1:8000/v1
```

## Requirements

- macOS on Apple silicon
- Rust/Cargo
- Python 3.10 or newer
- Git LFS, only if you choose the vendored GitHub model option

## Quick start

Clone only the code first. This avoids downloading large Git LFS model files
during `git clone`:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Dmitriy-Romanov/edge-lm-server.git
cd edge-lm-server
./run
```

`./run` builds the launcher and opens a small menu. The menu asks:

- where to load the model from: remote source or vendored GitHub LFS files
- which model to run
- which size to run

The remote option is the simplest first run. It does not require Git LFS, but it
does depend on the upstream model still being available.

The vendored GitHub option is the offline-safe path. It requires Git LFS, but it
downloads only the selected model and size instead of pulling every model file
in the repository.

## Models

The menu currently supports these remote model ids:

- `TheStageAI/gemma-4-E4B-it`
- `TheStageAI/gemma-4-E2B-it`

Remote size availability depends on what the upstream model repository provides.

This repository currently vendors both model ids:

- `TheStageAI/gemma-4-E4B-it`
- `TheStageAI/gemma-4-E2B-it`

For each vendored model, the launcher can use:

- `m`, the default size
- `l`, the larger size

For E4B, the selected size uses roughly:

- `m`: about 2.6 GB of model files
- `l`: about 3.1 GB of model files

For E2B, the selected size uses roughly:

- `m`: about 1.4 GB of model files
- `l`: about 1.7 GB of model files

Some shared files are also needed, such as the tokenizer, audio tower, and
vision tower. If those files are already present locally, the launcher skips the
Git LFS download on later runs.

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
      },
      {
        "id": "TheStageAI/gemma-4-E2B-it",
        "contextWindow": 128000,
        "maxTokens": 16000,
        "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 }
      }
    ]
  }
}
```

## Notes

The server uses Apple's MLX runtime through Edge-LM, so it is intended for
Apple silicon/macOS rather than Linux Docker.

See [NOTICE.md](NOTICE.md) for model attribution and terms.
Developer and maintainer notes live in [AGENTS.md](AGENTS.md).

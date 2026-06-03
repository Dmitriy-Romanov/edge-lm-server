# edge-lm-server launcher

Small Rust launcher for running an Edge-LM OpenAI-compatible local server for
Pi Agent.

This launcher uses models and the Python runtime package from
[TheStageAI/edge-lm](https://github.com/TheStageAI/edge-lm). The idea is simple:
the Edge-LM Gemma models were interesting enough to try locally, then useful
enough to wire into Pi Agent as a local coding model.

It creates a local runtime directory, checks Python/pip, creates a venv, installs
the Python dependencies, writes `server.py`, and starts the server.

## Build

```bash
cargo build --release
```

The binary will be:

```bash
target/release/edge-lm-server
```

## Run

```bash
./edge-lm-server
```

By default it uses:

- runtime dir: `.edge-lm-server`
- host: `127.0.0.1`
- port: `8000`
- model: `TheStageAI/gemma-4-E4B-it`
- model size: `m`
- context: `128000`

Useful flags:

```bash
./edge-lm-server --port 8001
./edge-lm-server --model TheStageAI/gemma-4-E2B-it --size m
./edge-lm-server --pi-models TheStageAI/gemma-4-E4B-it
./edge-lm-server --context 128000
./edge-lm-server setup
./edge-lm-server clean
```

Cleanup is simple: remove the binary and the runtime directory. The `clean`
command removes the runtime directory for you.

## License and model terms

This launcher is MIT licensed. It installs and uses
[TheStageAI/edge-lm](https://github.com/TheStageAI/edge-lm), which is also MIT
licensed. The Edge-LM model weights are derivatives of Google's Gemma models and
are additionally subject to the Gemma Terms of Use.

See [NOTICE.md](NOTICE.md) for attribution details.

The launcher prints a ready-to-paste provider block for
`~/.pi/agent/models.json`:

```json
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
```

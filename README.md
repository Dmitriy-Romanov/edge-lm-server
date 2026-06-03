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
./edge-lm-server setup --vendor-model
./edge-lm-server setup --preload-model
./edge-lm-server --offline
./edge-lm-server --prefer-remote
./edge-lm-server clean
```

## Vendored models

The launcher can use a model stored inside this repository. By default it looks
for:

```bash
models/TheStageAI/gemma-4-E4B-it
```

If that directory exists, it is used as the model source. Pi Agent still sees the
model id as `TheStageAI/gemma-4-E4B-it`; only the physical load path changes.

To download the default model into the repository:

```bash
cargo build --release
./target/release/edge-lm-server setup --vendor-model
```

To vendor another model:

```bash
./target/release/edge-lm-server setup \
  --model TheStageAI/gemma-4-E2B-it \
  --size m \
  --vendor-model
```

Large model files under `models/` are configured for Git LFS in
`.gitattributes`. Before committing vendored models, install and initialize Git
LFS:

```bash
brew install git-lfs
git lfs install
git add .gitattributes models/
git commit -m "Vendor Edge-LM model weights"
```

GitHub LFS rejects individual files over 2 GiB on free accounts. The large
`model_*.safetensors` files are therefore stored as `.part00`, `.part01`, ...
chunks. On startup, the launcher reassembles missing `model_*.safetensors` files
from those chunks before loading the model.

After a fresh clone on another Mac:

```bash
git lfs pull
cargo build --release
./target/release/edge-lm-server
```

If you want to try a newer upstream model version even when a vendored copy is
present, run with:

```bash
./target/release/edge-lm-server --prefer-remote
```

## Runtime cache

If you want to avoid depending on the model still being downloadable later,
preload it while the model is available:

```bash
cargo build --release
./target/release/edge-lm-server setup --preload-model
```

This installs the Python runtime and caches the selected model under:

```bash
.edge-lm-server/hf-home
```

After that, run the gateway in offline mode:

```bash
./target/release/edge-lm-server --offline
```

The `--offline` flag sets Hugging Face and Transformers offline environment
variables for the server process, so model loading uses the local cache.

To preload a different model or size:

```bash
./target/release/edge-lm-server setup \
  --model TheStageAI/gemma-4-E2B-it \
  --size m \
  --preload-model
```

## Docker note

The current server uses Apple's MLX runtime (`mlx.core`) through Edge-LM. MLX is
designed for Apple silicon and macOS/Metal, while normal Docker containers on a
Mac run Linux inside Docker Desktop's VM. That means a regular Docker image can
hold Python dependencies and cached files, but it is not a reliable autonomous
runtime for this MLX gateway.

For this repository, the practical offline path is therefore:

1. Vendor the model with `setup --vendor-model`.
2. Commit/push it through Git LFS.
3. Clone the repository and run the gateway normally on another Apple silicon
   Mac.

If a true Linux Docker container is required, the backend should be changed to a
Linux-compatible inference runtime such as llama.cpp, Ollama, or vLLM, using a
model format supported by that runtime. The Pi Agent provider can stay
OpenAI-compatible, but the model/runtime layer would be different from this MLX
implementation.

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

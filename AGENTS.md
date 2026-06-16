# AGENTS.md

Development notes for humans and coding agents working on this repository.

## Project purpose

`edge-lm-server` is a Python launcher plus FastAPI server for running
TheStageAI Edge-LM Gemma models locally as an OpenAI-compatible endpoint for Pi
Agent.

The root `./run` script:

- creates `.edge-lm-server/.venv`
- installs `git+https://github.com/TheStageAI/edge-lm.git`, `fastapi`, and
  `uvicorn` into that venv
- sets `PYTHONPATH=src`
- starts `edge_lm_server.cli menu`

The Python CLI:

- manages the interactive menu
- installs selected model files into `models/`
- starts the server with the selected model configuration

The Python server:

- loads Edge-LM through `edge_lm.models.load.load`
- uses MLX / `mlx_vlm.stream_generate`
- exposes `/v1/models`
- exposes streaming `/v1/chat/completions`
- adapts Pi Agent tool-call messages into the Gemma chat template shape

## Runtime assumptions

This project targets Apple silicon macOS. The inference path depends on MLX, so
a normal Linux Docker image is not a reliable runtime for this implementation.
Python-only packaging does not make the current backend cross-platform.

Potential backend split:

```text
MLX / Edge-LM safetensors -> Apple silicon macOS
GGUF / llama.cpp          -> macOS, Linux, Windows
```

The GGUF / llama.cpp path is not implemented. Do not expose it in the menu until
there is a real backend implementation.

## Model distribution strategy

Do not commit model weights to this repository.

GitHub LFS was removed as the model distribution path because public LFS
bandwidth can be exhausted and then installs fail for users. The GitHub
repository should contain code and docs only. Model files should come from the
upstream TheStageAI repositories on Hugging Face.

The default model is:

```bash
TheStageAI/gemma-4-E4B-it-qat
```

The smaller model is:

```bash
TheStageAI/gemma-4-E2B-it-qat
```

Only QAT models are exposed in the normal menu. The previous non-QAT models can
still be used manually with `--model`, but they are not the default user path.
TheStageAI also publishes GGUF artifacts for llama.cpp-compatible runtimes, but
this gateway currently uses native Edge-LM / MLX safetensors checkpoints. Do not
add GGUF to the normal menu unless a llama.cpp backend is implemented.

The install menu exposes four QAT install targets: E4B/E2B with size `m` or
`l`. The local start menu must show only variants that are actually installed
under `models/`.

The user-facing entry point is:

```bash
./run
```

That script bootstraps the venv and runs:

```bash
python -m edge_lm_server.cli menu
```

The menu has a dedicated "Show Pi Agent instructions" action. Normal server
startup should not print the provider JSON every time; keep that output in the
instructions action or explicit setup flows.

The menu supports two Hugging Face paths:

- remote startup, where Edge-LM downloads model files as needed
- install to `models/`, where the launcher uses `snapshot_download` with
  `allow_patterns` for the selected size and then starts from the local path in
  Hugging Face offline mode

The `models/` directory is ignored by git and is the portable local model
install location. A user can copy this directory to another Mac instead of
re-downloading model files.

## Useful launcher flags

```bash
--runtime-dir DIR
--host HOST
--port PORT
--model MODEL
--models-dir DIR
--pi-models LIST
--size SIZE
--context TOKENS
--reinstall
--install-model
--preload-model
--offline
--prefer-remote
```

`--install-model` downloads selected model files into `models/` by default. It
does not use Git LFS and should remain selective, not a full repository mirror.

`--preload-model` caches a model under `.edge-lm-server/hf-home`.

`--offline` sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for the Python
server process.

`--prefer-remote` tells the launcher to use the Hugging Face model id directly
even if local model files exist under `models/`.

Use `--reinstall` when upstream `TheStageAI/edge-lm` changed and the local venv
should be refreshed.

## Upstream sync notes

Before changing model ids or sizes, check upstream metadata without downloading
weights:

```bash
.edge-lm-server/.venv/bin/python - <<'PY'
from huggingface_hub import HfApi

for repo in [
    "TheStageAI/gemma-4-E4B-it-qat",
    "TheStageAI/gemma-4-E2B-it-qat",
]:
    info = HfApi().model_info(repo, files_metadata=True)
    print(repo, info.sha, info.last_modified)
    for sibling in info.siblings:
        if sibling.rfilename.endswith((".safetensors", ".json", ".jinja")):
            print(" ", sibling.rfilename, sibling.size)
PY
```

If the upstream package behavior changes, prefer a launcher/docs update over
vendoring weights.

## Verification

Run these after code changes:

```bash
python -m compileall src
./run --help
```

For packaging changes, also verify:

```bash
git status --short --branch
git lfs ls-files --size
```

`git lfs ls-files --size` should be empty after removing model weights from the
repository.

## License and terms

The launcher is MIT licensed. It installs and uses
`TheStageAI/edge-lm`, which is also MIT licensed. The model weights are
derivatives of Google's Gemma models and are additionally subject to the Gemma
Terms of Use. Keep [NOTICE.md](NOTICE.md) aligned with model changes.

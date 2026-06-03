# AGENTS.md

Development notes for humans and coding agents working on this repository.

## Project purpose

`edge-lm-server` is a small Rust launcher plus Python FastAPI server for running
TheStageAI Edge-LM Gemma models locally as an OpenAI-compatible endpoint for Pi
Agent.

The Rust binary:

- creates `.edge-lm-server`
- creates and manages a Python venv
- installs `git+https://github.com/TheStageAI/edge-lm.git`, `fastapi`, and
  `uvicorn`
- writes the bundled `src/server.py` into the runtime directory
- starts the server with the selected model configuration
- supports an interactive `menu` action used by the root `./run` script

The Python server:

- loads Edge-LM through `edge_lm.models.load.load`
- uses MLX / `mlx_vlm.stream_generate`
- exposes `/v1/models`
- exposes streaming `/v1/chat/completions`
- adapts Pi Agent tool-call messages into the Gemma chat template shape

## Runtime assumptions

This project targets Apple silicon macOS. The inference path depends on MLX, so
a normal Linux Docker image is not a reliable runtime for this implementation.

## Vendored model strategy

The default model is:

```bash
TheStageAI/gemma-4-E4B-it
```

The repository currently vendors:

```bash
models/TheStageAI/gemma-4-E4B-it
models/TheStageAI/gemma-4-E2B-it
```

If that directory exists and `--prefer-remote` is not set, the launcher passes
the local path as `EDGE_LM_MODEL_SOURCE`. It still passes the public model id as
`EDGE_LM_MODEL_ID`, so Pi Agent sees the normal model id.

If `--prefer-remote` is set, vendored files are ignored and the model id is used
directly as the load source.

The user-facing entry point is:

```bash
./run
```

That script builds the release binary and runs:

```bash
./target/release/edge-lm-server menu
```

The README intentionally recommends:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone ...
```

This prevents an installed Git LFS smudge filter from downloading every model
file during clone. The menu can then selectively download only the selected
vendored size.

## Split model files

GitHub LFS rejects individual files over 2 GiB on free accounts. For E4B, the
full `model_l.safetensors` and `model_m.safetensors` files are therefore not
tracked directly. They are stored as LFS chunks:

```bash
models/TheStageAI/gemma-4-E4B-it/model_l.safetensors.part00
models/TheStageAI/gemma-4-E4B-it/model_l.safetensors.part01
models/TheStageAI/gemma-4-E4B-it/model_m.safetensors.part00
models/TheStageAI/gemma-4-E4B-it/model_m.safetensors.part01
```

On startup, `src/main.rs` calls `restore_split_vendored_models`. If the target
`.safetensors` file is missing and matching `.partNN` files exist, the launcher
reassembles the file before starting Python.

The restored full files are ignored by git:

```bash
models/TheStageAI/gemma-4-E4B-it/model_*.safetensors
```

E2B model files are below the GitHub LFS per-object limit, so
`model_m.safetensors` and `model_l.safetensors` are tracked directly through
Git LFS.

The interactive menu uses `git lfs pull --include ... --exclude ""` to download
only the files needed for the selected model and size. For E4B `m`, it includes:

- shared `config.json`
- shared `audio_tower.safetensors`
- shared `vision_tower.safetensors`
- shared `tokenizer.json`
- shared `tokenizer_config.json`
- shared `chat_template.jinja`
- `ple_m.safetensors`
- `model_m.safetensors.part00`
- `model_m.safetensors.part01`

For E4B `l`, it swaps the `m` files for `ple_l` and `model_l` parts. For E2B,
it pulls the full `model_m.safetensors` or `model_l.safetensors` file instead
of split parts.

The launcher checks whether selected files are already present and real LFS
content, not pointer files, before pulling. This protects repeat runs from
re-downloading the same model files.

Vendored runs automatically set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.
This is intentional: starting a downloaded model must not silently fetch missing
tokenizer/template files from Hugging Face. If a vendored model is incomplete,
the launcher should fail or pull the selected files explicitly through the menu.

## Updating vendored models

Use this only when intentionally refreshing model files in the repository.

Install Git LFS:

```bash
brew install git-lfs
git lfs install
```

Download the default model into `models/`:

```bash
cargo build --release
./target/release/edge-lm-server setup --vendor-model
```

Download another model:

```bash
./target/release/edge-lm-server setup \
  --model TheStageAI/gemma-4-E2B-it \
  --size m \
  --vendor-model
```

If any generated model file is larger than GitHub LFS allows, split it before
committing:

```bash
split -b 1900m -d -a 2 \
  models/TheStageAI/gemma-4-E4B-it/model_m.safetensors \
  models/TheStageAI/gemma-4-E4B-it/model_m.safetensors.part
```

Repeat for other files over the limit. Keep the original full file locally, but
remove it from the git index:

```bash
git rm --cached models/TheStageAI/gemma-4-E4B-it/model_m.safetensors
```

Then add the chunks and metadata:

```bash
git add .gitattributes .gitignore models/
git commit -m "Vendor split Edge-LM model weights"
git lfs push origin main
git push origin main
```

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
--preload-model
--vendor-model
--offline
--prefer-remote
```

`--preload-model` caches a model under `.edge-lm-server/hf-home`.

`--offline` sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for the Python
server process.

`--vendor-model` downloads model files into `models/`; use it as a maintainer
operation, not as a normal user workflow.

## Verification

Run these after code changes:

```bash
cargo fmt --check
cargo check
cargo run -- --help
```

For model packaging changes, also verify:

```bash
git lfs ls-files --size
git status --short --branch
```

No reachable LFS object should exceed GitHub's per-object size limit.

## License and terms

The launcher is MIT licensed. It installs and uses
`TheStageAI/edge-lm`, which is also MIT licensed. The model weights are
derivatives of Google's Gemma models and are additionally subject to the Gemma
Terms of Use. Keep [NOTICE.md](NOTICE.md) aligned with model changes.

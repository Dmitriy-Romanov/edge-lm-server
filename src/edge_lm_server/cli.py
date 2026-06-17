from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODELS_DIR = Path("models")
DEFAULT_RUNTIME_DIR = Path(".edge-lm-server")
DEFAULT_MODEL = "TheStageAI/gemma-4-E4B-it-qat"
SMALLER_MODEL = "TheStageAI/gemma-4-E2B-it-qat"
DEFAULT_PI_MODELS = "local-edge-e4b-m,local-edge-e4b-l,local-edge-e2b-m,local-edge-e2b-l"


@dataclass(frozen=True)
class ModelOption:
    id: str
    name: str
    description: str
    sizes: dict[str, str]


MODEL_OPTIONS = [
    ModelOption(
        id=DEFAULT_MODEL,
        name="E4B",
        description="larger QAT model",
        sizes={"m": "about 3.1 GB", "l": "about 3.7 GB"},
    ),
    ModelOption(
        id=SMALLER_MODEL,
        name="E2B",
        description="smaller QAT model",
        sizes={"m": "about 1.8 GB", "l": "about 2.1 GB"},
    ),
]


@dataclass
class Config:
    action: str
    runtime_dir: Path
    host: str
    port: int
    model: str
    pi_models: list[str]
    models_dir: Path
    size: str
    context_tokens: int
    reinstall: bool
    install_model: bool
    preload_model: bool
    offline: bool
    prefer_remote: bool


@dataclass(frozen=True)
class LocalModel:
    model: str
    size: str


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv or sys.argv[1:]))
    except (KeyboardInterrupt, EOFError):
        print()
        print("Cancelled.")
        return 130


def run(config: Config) -> int:
    if config.action == "clean":
        clean_runtime(config.runtime_dir)
        return 0

    if config.action == "menu":
        configure_from_menu(config)

    if config.install_model:
        install_model_files(config)

    if config.preload_model:
        preload_model(config)

    if config.action == "setup":
        print_pi_config(config)
        print(f"setup complete: {config.runtime_dir}")
        return 0

    run_server(config)
    return 0


def parse_args(argv: list[str]) -> Config:
    parser = argparse.ArgumentParser(prog="edge-lm-server")
    parser.add_argument("action", nargs="?", choices=["run", "menu", "setup", "clean"], default="run")
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--pi-models", default=DEFAULT_PI_MODELS)
    parser.add_argument("--size", default="m")
    parser.add_argument("--context", type=int, default=128_000)
    parser.add_argument("--reinstall", action="store_true")
    parser.add_argument("--install-model", action="store_true")
    parser.add_argument("--preload-model", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--prefer-remote", action="store_true")
    args = parser.parse_args(argv)

    pi_models = [item.strip() for item in args.pi_models.split(",") if item.strip()]
    if not pi_models:
        parser.error("--pi-models must contain at least one model id")

    return Config(
        action=args.action,
        runtime_dir=args.runtime_dir.resolve(),
        host=args.host,
        port=args.port,
        model=args.model,
        pi_models=pi_models,
        models_dir=args.models_dir.resolve(),
        size=args.size,
        context_tokens=args.context,
        reinstall=args.reinstall,
        install_model=args.install_model,
        preload_model=args.preload_model,
        offline=args.offline,
        prefer_remote=args.prefer_remote,
    )


def configure_from_menu(config: Config) -> None:
    print("edge-lm-server setup")
    print()

    downloaded = downloaded_local_models(config)
    if not downloaded:
        print("No local models installed.")
        config.prefer_remote = False
        config.install_model = True
        config.offline = True
        choose_install_target(config)
        config.pi_models = [model_alias(config.model, config.size)]
        print()
        print(
            f"Starting {model_alias(config.model, config.size)} "
            f"({config.model}, size {config.size}) from {model_source_label(config)}."
        )
        print()
        return

    print("Local model files:")
    for index, item in enumerate(downloaded, start=1):
        print(f"  {index}) {local_model_label(item.model, item.size)} ({model_storage_label(item.model, item.size)})")

    print()
    print("Choose what to do:")
    print("  1) Start server with local model files")
    print("  2) Show Pi Agent instructions")
    print("  3) Download/install selected model into models/")
    action = prompt_choice("Action", ["1", "2", "3"], "1")

    if action == "1":
        selected = choose_local_model(downloaded)
        config.model = selected.model
        config.size = selected.size
        config.prefer_remote = False
    elif action == "2":
        configure_pi_instructions_from_menu(config)
        print_pi_config(config)
        raise SystemExit(0)
    elif action == "3":
        config.prefer_remote = False
        config.install_model = True
        config.offline = True
        choose_install_target(config)

    config.pi_models = [model_alias(config.model, config.size)]
    print()
    print(
        f"Starting {model_alias(config.model, config.size)} "
        f"({config.model}, size {config.size}) from {model_source_label(config)}."
    )
    print()


def choose_local_model(downloaded: list[LocalModel]) -> LocalModel:
    if len(downloaded) == 1:
        item = downloaded[0]
        print()
        print(f"Using local model files: {local_model_label(item.model, item.size)} ({model_storage_label(item.model, item.size)})")
        return item

    print()
    print("Choose local model files:")
    for index, item in enumerate(downloaded, start=1):
        print(f"  {index}) {local_model_label(item.model, item.size)} ({model_storage_label(item.model, item.size)})")
    choice = prompt_choice("Local model", numbered_choices(len(downloaded)), "1")
    return downloaded[int(choice) - 1]


def choose_model(config: Config, show_download_sizes: bool) -> None:
    print()
    print("Choose model:")
    for index, option in enumerate(MODEL_OPTIONS, start=1):
        if show_download_sizes:
            print(
                f"  {index}) {option.id} ({option.name}, {option.description}, "
                f"download {option.sizes['m']})"
            )
        else:
            print(f"  {index}) {option.id} ({option.name}, {option.description})")
    choice = prompt_choice("Model", numbered_choices(len(MODEL_OPTIONS)), "1")
    config.model = MODEL_OPTIONS[int(choice) - 1].id


def choose_install_target(config: Config) -> None:
    print()
    print("Choose model to install:")
    targets = model_variants()
    for index, (option, size) in enumerate(targets, start=1):
        print(
            f"  {index}) {model_alias(option.id, size)} -> {option.id} "
            f"({option.name}, {option.description}, "
            f"size {size}, download {option.sizes[size]})"
        )
    choice = prompt_choice("Install model", numbered_choices(len(targets)), "1")
    option, size = targets[int(choice) - 1]
    config.model = option.id
    config.size = size


def configure_pi_instructions_from_menu(config: Config) -> None:
    print()
    print("Choose model to show in Pi Agent config:")
    variants = model_variants()
    for index, (option, size) in enumerate(variants, start=1):
        print(
            f"  {index}) {model_alias(option.id, size)} "
            f"({option.name}, {option.description}, size {size})"
        )
    all_choice = str(len(variants) + 1)
    print(f"  {all_choice}) All local aliases")
    choice = prompt_choice("Pi Agent model", numbered_choices(len(variants) + 1), all_choice)
    if choice == all_choice:
        config.pi_models = [model_alias(option.id, size) for option, size in variants]
    else:
        option, size = variants[int(choice) - 1]
        config.pi_models = [model_alias(option.id, size)]


def prompt_choice(prompt: str, allowed: list[str], default: str) -> str:
    while True:
        value = input(f"{prompt} [{default}]: ").strip() or default
        if value in allowed:
            return value
        print(f"Please choose one of: {', '.join(allowed)}")


def numbered_choices(count: int) -> list[str]:
    return [str(index) for index in range(1, count + 1)]


def downloaded_local_models(config: Config) -> list[LocalModel]:
    downloaded = []
    for option, size in model_variants():
        candidate = copy_config(config)
        candidate.model = option.id
        candidate.size = size
        candidate.prefer_remote = False
        if local_model_ready(candidate):
            downloaded.append(LocalModel(model=option.id, size=size))
    return downloaded


def copy_config(config: Config) -> Config:
    return Config(**config.__dict__)


def model_menu_label(model: str) -> str:
    option = model_option(model)
    if option is None:
        return model
    return f"{option.id} ({option.name}, {option.description})"


def local_model_label(model: str, size: str) -> str:
    return f"{model_alias(model, size)} -> {model_menu_label(model)}"


def model_storage_label(model: str, size: str) -> str:
    storage = model_size_download_label(model, size)
    if size == "m":
        return f"{storage} installed"
    return f"{storage} installed, Edge-LM size {size}"


def model_size_download_label(model: str, size: str) -> str:
    option = model_option(model)
    if option is None:
        return "unknown size"
    return option.sizes.get(size, "unknown size")


def model_alias(model: str, size: str) -> str:
    option = model_option(model)
    if option is None:
        return f"local-edge-{size}"
    return f"local-edge-{option.name.lower()}-{size}"


def model_variants() -> list[tuple[ModelOption, str]]:
    return [(option, size) for option in MODEL_OPTIONS for size in ("m", "l")]


def model_option(model: str) -> ModelOption | None:
    return next((option for option in MODEL_OPTIONS if option.id == model), None)


def model_source_label(config: Config) -> str:
    if config.prefer_remote:
        if config.preload_model:
            return "Hugging Face cache"
        return "Hugging Face remote source"
    return "local model files"


def install_model_files(config: Config) -> None:
    if model_looks_like_path(config.model):
        raise SystemExit("--install-model expects --model to be a Hugging Face repo id")

    destination = local_model_path(config)
    if local_model_ready(config):
        print(f"local model files already exist: {destination} ({config.size})")
        return

    destination.mkdir(parents=True, exist_ok=True)
    patterns = model_install_patterns(config.size)
    print(f"downloading model files into {destination}: {config.model} ({config.size})")

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=config.model,
        local_dir=destination,
        allow_patterns=patterns,
    )
    print(f"installed {config.model} into {destination}")

    if not local_model_ready(config):
        raise SystemExit(f"download finished, but local model files are incomplete: {destination}")


def model_install_patterns(size: str) -> list[str]:
    return [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "audio_tower.safetensors",
        "vision_tower.safetensors",
        f"model_{size}.safetensors",
        f"ple_{size}.safetensors",
        f"summary_{size}.json",
    ]


def local_model_ready(config: Config) -> bool:
    local_model = local_model_path(config)
    if not local_model.exists():
        return False

    model = local_model / f"model_{config.size}.safetensors"
    model_ready = real_file_at_least(model, 100 * 1024 * 1024)
    required = [
        local_model / "config.json",
        local_model / "tokenizer_config.json",
        local_model / f"ple_{config.size}.safetensors",
        local_model / "audio_tower.safetensors",
        local_model / "vision_tower.safetensors",
        local_model / "tokenizer.json",
    ]
    return model_ready and all(real_file_at_least(path, min_size(path)) for path in required)


def min_size(path: Path) -> int:
    if path.name.endswith(".json"):
        return 1024
    if path.name == "tokenizer.json":
        return 1024 * 1024
    return 100 * 1024 * 1024


def real_file_at_least(path: Path, minimum: int) -> bool:
    try:
        return path.stat().st_size >= minimum and not is_lfs_pointer(path)
    except FileNotFoundError:
        return False


def is_lfs_pointer(path: Path) -> bool:
    try:
        return path.read_bytes()[:128].startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


def preload_model(config: Config) -> None:
    model_source = resolve_model_source(config)
    print(f"preloading model into local cache: {config.model} ({config.size}) from {model_source}")
    code = (
        "from edge_lm.models.load import load\n"
        "import os\n"
        "load(os.environ['EDGE_LM_MODEL_SOURCE'], size=os.environ['EDGE_LM_SIZE'])\n"
        "print('cached')\n"
    )
    env = runtime_env(config, model_source)
    subprocess.run([sys.executable, "-c", code], check=True, env=env, cwd=config.runtime_dir)


def run_server(config: Config) -> None:
    model_source = resolve_model_source(config)
    base_url = f"http://{config.host}:{config.port}"
    status_url = f"{base_url}/status"
    print(
        f"starting server at {terminal_link(base_url)} using "
        f"{model_alias(config.model, config.size)} ({config.model}, size {config.size}) "
        f"from {model_source}"
    )
    print(f"status dashboard: {terminal_link(status_url)}")
    print(f"runtime: {config.runtime_dir}")
    env = runtime_env(config, model_source)
    subprocess.run([sys.executable, "-m", "edge_lm_server.server"], check=True, env=env, cwd=Path.cwd())


def terminal_link(url: str, label: str | None = None) -> str:
    text = label or url
    return f"\033]8;;{url}\033\\{text}\033]8;;\033\\"


def runtime_env(config: Config, model_source: Path | str) -> dict[str, str]:
    config.runtime_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    python_path = str(Path.cwd() / "src")
    env["PYTHONPATH"] = python_path if not env.get("PYTHONPATH") else f"{python_path}{os.pathsep}{env['PYTHONPATH']}"
    env["EDGE_LM_MODEL_SOURCE"] = str(model_source)
    env["EDGE_LM_MODEL_ID"] = model_alias(config.model, config.size)
    env["EDGE_LM_SIZE"] = config.size
    env["EDGE_LM_CONTEXT_TOKENS"] = str(config.context_tokens)
    env["EDGE_LM_HOST"] = config.host
    env["EDGE_LM_PORT"] = str(config.port)
    env["HF_HOME"] = str(config.runtime_dir / "hf-home")
    env["TRANSFORMERS_CACHE"] = str(config.runtime_dir / "hf-home" / "transformers")
    if config.offline or not config.prefer_remote:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    return env


def resolve_model_source(config: Config) -> Path | str:
    if config.prefer_remote:
        return config.model
    if model_looks_like_path(config.model):
        return Path(config.model).resolve()
    local_model = local_model_path(config)
    if local_model_ready(config):
        return local_model
    return config.model


def local_model_path(config: Config) -> Path:
    path = config.models_dir
    for part in config.model.split("/"):
        if not part or part in {".", ".."}:
            raise SystemExit(f"invalid model id for local model path: {config.model}")
        path = path / part
    return path


def model_looks_like_path(value: str) -> bool:
    return value.startswith((".", "/")) or "\\" in value or Path(value).exists()


def print_pi_config(config: Config) -> None:
    print()
    print("Repository:")
    print("https://github.com/Dmitriy-Romanov/edge-lm-server")
    print()
    print("Quick start:")
    print("git clone https://github.com/Dmitriy-Romanov/edge-lm-server.git")
    print("cd edge-lm-server")
    print("./run")
    print()
    print("Add this provider to ~/.pi/agent/models.json:")
    print()
    provider = {
        "baseUrl": f"http://{config.host}:{config.port}/v1",
        "api": "openai-completions",
        "apiKey": "local-key",
        "compat": {
            "supportsDeveloperRole": False,
            "supportsReasoningEffort": False,
            "supportsUsageInStreaming": True,
        },
        "models": [
            {
                "id": model,
                "contextWindow": config.context_tokens,
                "maxTokens": 16000,
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            }
            for model in config.pi_models
        ],
    }
    print('    "local-edge": ' + json.dumps(provider, indent=2).replace("\n", "\n    "))
    print()
    print(
        "The server process will expose the selected local alias. "
        "One running server process serves one selected model alias at a time."
    )
    print()


def clean_runtime(runtime_dir: Path) -> None:
    if not runtime_dir.exists():
        print(f"nothing to clean: {runtime_dir}")
        return
    shutil.rmtree(runtime_dir)
    print(f"removed {runtime_dir}")


if __name__ == "__main__":
    raise SystemExit(main())

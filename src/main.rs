use std::collections::BTreeMap;
use std::env;
use std::ffi::{OsStr, OsString};
use std::fs;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};

const EDGE_LM_REPO: &str = "git+https://github.com/TheStageAI/edge-lm.git";
const SERVER_PY: &str = include_str!("server.py");
const DEFAULT_MODELS_DIR: &str = "models";
const PYTHON_CANDIDATES: &[&str] = &[
    "python3.14",
    "python3.13",
    "python3.12",
    "python3.11",
    "python3.10",
    "python3",
];
const BREW_PYTHON_FORMULAE: &[&str] = &[
    "python@3.14",
    "python@3.13",
    "python@3.12",
    "python@3.11",
    "python@3.10",
    "python",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Action {
    Run,
    Setup,
    Clean,
    Help,
}

#[derive(Debug, Clone)]
struct Config {
    action: Action,
    runtime_dir: PathBuf,
    host: String,
    port: u16,
    model: String,
    pi_models: Vec<String>,
    models_dir: PathBuf,
    size: String,
    context_tokens: usize,
    reinstall: bool,
    preload_model: bool,
    vendor_model: bool,
    offline: bool,
    prefer_remote: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            action: Action::Run,
            runtime_dir: PathBuf::from(".edge-lm-server"),
            host: "127.0.0.1".to_string(),
            port: 8000,
            model: "TheStageAI/gemma-4-E4B-it".to_string(),
            pi_models: vec![
                "TheStageAI/gemma-4-E4B-it".to_string(),
                "TheStageAI/gemma-4-E2B-it".to_string(),
            ],
            models_dir: PathBuf::from(DEFAULT_MODELS_DIR),
            size: "m".to_string(),
            context_tokens: 128_000,
            reinstall: false,
            preload_model: false,
            vendor_model: false,
            offline: false,
            prefer_remote: false,
        }
    }
}

fn main() {
    if let Err(err) = real_main() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn real_main() -> Result<(), String> {
    let config = parse_args(env::args().skip(1).collect())?;

    if config.action == Action::Help {
        print_help();
        return Ok(());
    }

    let runtime_dir = absolutize(&config.runtime_dir)?;

    if config.action == Action::Clean {
        clean_runtime(&runtime_dir)?;
        return Ok(());
    }

    restore_split_vendored_models(&config)?;

    fs::create_dir_all(&runtime_dir)
        .map_err(|e| format!("failed to create {}: {e}", runtime_dir.display()))?;

    let python = ensure_python()?;
    let git = ensure_git()?;
    let venv_python = runtime_dir.join(".venv").join("bin").join("python");

    ensure_venv(&python, &venv_python, &runtime_dir)?;
    ensure_pip(&venv_python)?;
    install_dependencies(&venv_python, &git, &runtime_dir, config.reinstall)?;
    write_server(&runtime_dir)?;
    if config.vendor_model {
        vendor_model(&venv_python, &config)?;
    }
    if config.preload_model {
        preload_model(&venv_python, &runtime_dir, &config)?;
    }
    print_pi_config(&config);

    if config.action == Action::Setup {
        println!("setup complete: {}", runtime_dir.display());
        return Ok(());
    }

    run_server(&venv_python, &runtime_dir, &config)
}

fn parse_args(args: Vec<String>) -> Result<Config, String> {
    let mut config = Config::default();
    let mut i = 0;

    while i < args.len() {
        match args[i].as_str() {
            "run" => config.action = Action::Run,
            "setup" => config.action = Action::Setup,
            "clean" => config.action = Action::Clean,
            "-h" | "--help" | "help" => config.action = Action::Help,
            "--runtime-dir" => {
                i += 1;
                config.runtime_dir = PathBuf::from(value_after(&args, i, "--runtime-dir")?);
            }
            "--host" => {
                i += 1;
                config.host = value_after(&args, i, "--host")?;
            }
            "--port" => {
                i += 1;
                config.port = value_after(&args, i, "--port")?
                    .parse()
                    .map_err(|_| "--port must be a number".to_string())?;
            }
            "--model" => {
                i += 1;
                config.model = value_after(&args, i, "--model")?;
            }
            "--models-dir" => {
                i += 1;
                config.models_dir = PathBuf::from(value_after(&args, i, "--models-dir")?);
            }
            "--pi-models" => {
                i += 1;
                config.pi_models = value_after(&args, i, "--pi-models")?
                    .split(',')
                    .map(str::trim)
                    .filter(|item| !item.is_empty())
                    .map(str::to_string)
                    .collect();
                if config.pi_models.is_empty() {
                    return Err("--pi-models must contain at least one model id".to_string());
                }
            }
            "--size" => {
                i += 1;
                config.size = value_after(&args, i, "--size")?;
            }
            "--context" => {
                i += 1;
                config.context_tokens = value_after(&args, i, "--context")?
                    .parse()
                    .map_err(|_| "--context must be a number".to_string())?;
            }
            "--reinstall" => config.reinstall = true,
            "--preload-model" => config.preload_model = true,
            "--vendor-model" => config.vendor_model = true,
            "--offline" => config.offline = true,
            "--prefer-remote" => config.prefer_remote = true,
            other => return Err(format!("unknown argument: {other}")),
        }
        i += 1;
    }

    Ok(config)
}

fn value_after(args: &[String], index: usize, flag: &str) -> Result<String, String> {
    args.get(index)
        .cloned()
        .ok_or_else(|| format!("{flag} needs a value"))
}

fn print_help() {
    println!(
        "edge-lm-server\n\n\
         Usage:\n\
           edge-lm-server [run] [options]\n\
           edge-lm-server setup [options]\n\
           edge-lm-server clean [--runtime-dir DIR]\n\n\
         Options:\n\
           --runtime-dir DIR   Runtime directory, default .edge-lm-server\n\
           --host HOST         Bind host, default 127.0.0.1\n\
           --port PORT         Bind port, default 8000\n\
           --model MODEL       Hugging Face model id\n\
           --models-dir DIR    Vendored models directory, default models\n\
           --pi-models LIST    Comma-separated Pi model ids, default E4B,E2B\n\
           --size SIZE         Edge-LM size, default m\n\
           --context TOKENS    Context window, default 128000\n\
           --reinstall         Reinstall Python dependencies\n\
           --preload-model     Download/cache the selected model during setup/run\n\
           --vendor-model      Download the selected model into --models-dir\n\
           --offline           Use local model/dependency caches only while running\n\
           --prefer-remote     Ignore vendored model directory and use --model directly\n\
           -h, --help          Show this help"
    );
}

fn absolutize(path: &Path) -> Result<PathBuf, String> {
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    env::current_dir()
        .map(|cwd| cwd.join(path))
        .map_err(|e| format!("failed to read current directory: {e}"))
}

fn ensure_python() -> Result<PathBuf, String> {
    if let Ok(path) = env::var("EDGE_LM_PYTHON") {
        let candidate = PathBuf::from(path);
        if python_is_supported(&candidate) {
            return Ok(candidate);
        }
        return Err("EDGE_LM_PYTHON is set, but it is not Python >= 3.10".to_string());
    }

    if let Some(path) = find_supported_python(None) {
        return Ok(path);
    }

    let brew = find_on_path("brew").ok_or_else(|| {
        "Python >= 3.10 was not found. Install Homebrew from https://brew.sh, then rerun this launcher.".to_string()
    })?;

    println!("Python >= 3.10 not found; installing Python with Homebrew...");
    run_inherit(Command::new(&brew).arg("install").arg("python"))?;

    if let Some(path) = find_supported_python(Some(&brew)) {
        return Ok(path);
    }

    Err(
        "Homebrew finished, but Python >= 3.10 still was not found. Try adding Homebrew to PATH or set EDGE_LM_PYTHON=/path/to/python3."
            .to_string(),
    )
}

fn find_supported_python(brew: Option<&Path>) -> Option<PathBuf> {
    for name in PYTHON_CANDIDATES {
        if let Some(path) = find_on_path(name) {
            if python_is_supported(&path) {
                return Some(path);
            }
        }
    }

    if let Some(brew) = brew {
        for formula in BREW_PYTHON_FORMULAE {
            if let Some(prefix) = brew_prefix(brew, formula) {
                for name in PYTHON_CANDIDATES {
                    let path = prefix.join("bin").join(name);
                    if python_is_supported(&path) {
                        return Some(path);
                    }
                }
            }
        }
    }

    None
}

fn brew_prefix(brew: &Path, formula: &str) -> Option<PathBuf> {
    let output = Command::new(brew)
        .arg("--prefix")
        .arg(formula)
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let prefix = String::from_utf8(output.stdout).ok()?;
    let prefix = prefix.trim();
    if prefix.is_empty() {
        None
    } else {
        Some(PathBuf::from(prefix))
    }
}

fn ensure_git() -> Result<PathBuf, String> {
    if let Some(path) = find_on_path("git") {
        return Ok(path);
    }

    let brew = find_on_path("brew").ok_or_else(|| {
        "git was not found. Install Homebrew from https://brew.sh, then rerun this launcher."
            .to_string()
    })?;

    println!("git not found; installing git with Homebrew...");
    run_inherit(Command::new(&brew).arg("install").arg("git"))?;

    find_on_path("git")
        .ok_or_else(|| "Homebrew finished, but git still was not found on PATH".to_string())
}

fn python_is_supported(path: &Path) -> bool {
    let code = "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)";
    Command::new(path)
        .arg("-c")
        .arg(code)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn ensure_venv(python: &Path, venv_python: &Path, runtime_dir: &Path) -> Result<(), String> {
    if venv_python.exists() {
        return Ok(());
    }

    println!("creating venv: {}", runtime_dir.join(".venv").display());
    run_inherit(
        Command::new(python)
            .arg("-m")
            .arg("venv")
            .arg(runtime_dir.join(".venv")),
    )
}

fn ensure_pip(venv_python: &Path) -> Result<(), String> {
    if run_quiet(
        Command::new(venv_python)
            .arg("-m")
            .arg("pip")
            .arg("--version"),
    ) {
        return Ok(());
    }

    println!("pip not found in venv; trying ensurepip...");
    run_inherit(
        Command::new(venv_python)
            .arg("-m")
            .arg("ensurepip")
            .arg("--upgrade"),
    )?;

    if run_quiet(
        Command::new(venv_python)
            .arg("-m")
            .arg("pip")
            .arg("--version"),
    ) {
        Ok(())
    } else {
        Err("pip is still unavailable in the venv".to_string())
    }
}

fn install_dependencies(
    venv_python: &Path,
    git: &Path,
    runtime_dir: &Path,
    reinstall: bool,
) -> Result<(), String> {
    let marker = runtime_dir.join(".deps-installed");
    if marker.exists() && !reinstall {
        return Ok(());
    }

    println!("installing Python dependencies...");
    let pip_path = path_with_command_parent(git);
    run_inherit(
        Command::new(venv_python)
            .env("PATH", &pip_path)
            .arg("-m")
            .arg("pip")
            .arg("install")
            .arg("--no-cache-dir")
            .arg("--upgrade")
            .arg("pip")
            .arg("setuptools")
            .arg("wheel"),
    )?;

    run_inherit(
        Command::new(venv_python)
            .env("PATH", &pip_path)
            .arg("-m")
            .arg("pip")
            .arg("install")
            .arg("--no-cache-dir")
            .arg("--upgrade")
            .arg(EDGE_LM_REPO)
            .arg("fastapi")
            .arg("uvicorn"),
    )?;

    fs::write(&marker, "ok\n").map_err(|e| format!("failed to write {}: {e}", marker.display()))
}

fn write_server(runtime_dir: &Path) -> Result<(), String> {
    let server_path = runtime_dir.join("server.py");
    fs::write(&server_path, SERVER_PY)
        .map_err(|e| format!("failed to write {}: {e}", server_path.display()))?;
    Ok(())
}

fn vendor_model(venv_python: &Path, config: &Config) -> Result<(), String> {
    if model_looks_like_path(&config.model) {
        return Err("--vendor-model expects --model to be a Hugging Face repo id".to_string());
    }

    let destination = vendored_model_path(config)?;
    fs::create_dir_all(&destination)
        .map_err(|e| format!("failed to create {}: {e}", destination.display()))?;

    println!(
        "downloading model into repository: {} -> {}",
        config.model,
        destination.display()
    );

    let code = "\
from huggingface_hub import snapshot_download\n\
import os\n\
repo_id = os.environ['EDGE_LM_MODEL_ID']\n\
local_dir = os.environ['EDGE_LM_VENDOR_DIR']\n\
snapshot_download(repo_id=repo_id, local_dir=local_dir)\n\
print(f'vendored {repo_id} into {local_dir}')\n";

    run_inherit(
        Command::new(venv_python)
            .arg("-c")
            .arg(code)
            .env("EDGE_LM_MODEL_ID", &config.model)
            .env("EDGE_LM_VENDOR_DIR", &destination),
    )
}

fn restore_split_vendored_models(config: &Config) -> Result<(), String> {
    if config.prefer_remote || model_looks_like_path(&config.model) {
        return Ok(());
    }

    let vendored = vendored_model_path(config)?;
    if !vendored.exists() {
        return Ok(());
    }

    let mut groups: BTreeMap<PathBuf, Vec<PathBuf>> = BTreeMap::new();
    let entries = fs::read_dir(&vendored)
        .map_err(|e| format!("failed to read {}: {e}", vendored.display()))?;

    for entry in entries {
        let entry = entry.map_err(|e| format!("failed to read {}: {e}", vendored.display()))?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        let Some(file_name) = path.file_name().and_then(OsStr::to_str) else {
            continue;
        };
        let Some(base_name) = split_part_base_name(file_name) else {
            continue;
        };
        groups
            .entry(vendored.join(base_name))
            .or_default()
            .push(path);
    }

    for (target, mut parts) in groups {
        if target.exists() {
            continue;
        }

        parts.sort();
        println!(
            "reassembling vendored model file: {} from {} parts",
            target.display(),
            parts.len()
        );

        let mut output = File::create(&target)
            .map_err(|e| format!("failed to create {}: {e}", target.display()))?;
        for part in parts {
            let mut input =
                File::open(&part).map_err(|e| format!("failed to open {}: {e}", part.display()))?;
            io::copy(&mut input, &mut output)
                .map_err(|e| format!("failed to append {}: {e}", part.display()))?;
        }
    }

    Ok(())
}

fn split_part_base_name(file_name: &str) -> Option<&str> {
    let (base_name, part_number) = file_name.rsplit_once(".part")?;
    if base_name.is_empty()
        || part_number.is_empty()
        || !part_number.chars().all(|ch| ch.is_ascii_digit())
    {
        return None;
    }
    Some(base_name)
}

fn preload_model(venv_python: &Path, runtime_dir: &Path, config: &Config) -> Result<(), String> {
    let model_source = model_source(config)?;
    println!(
        "preloading model into local cache: {} ({}) from {}",
        config.model,
        config.size,
        model_source.display()
    );

    let code = "\
from edge_lm.models.load import load\n\
import os\n\
model = os.environ['EDGE_LM_MODEL_SOURCE']\n\
size = os.environ['EDGE_LM_SIZE']\n\
load(model, size=size)\n\
print(f'cached {model} ({size})')\n";

    run_inherit(
        Command::new(venv_python)
            .arg("-c")
            .arg(code)
            .current_dir(runtime_dir)
            .env("EDGE_LM_MODEL_SOURCE", &model_source)
            .env("EDGE_LM_SIZE", &config.size)
            .env("HF_HOME", runtime_dir.join("hf-home"))
            .env(
                "TRANSFORMERS_CACHE",
                runtime_dir.join("hf-home").join("transformers"),
            ),
    )
}

fn run_server(venv_python: &Path, runtime_dir: &Path, config: &Config) -> Result<(), String> {
    let model_source = model_source(config)?;
    println!(
        "starting server at http://{}:{} using {} ({}) from {}",
        config.host,
        config.port,
        config.model,
        config.size,
        model_source.display()
    );
    println!("runtime: {}", runtime_dir.display());

    let mut command = Command::new(venv_python);
    command
        .arg(runtime_dir.join("server.py"))
        .current_dir(runtime_dir)
        .env("EDGE_LM_MODEL_SOURCE", &model_source)
        .env("EDGE_LM_MODEL_ID", &config.model)
        .env("EDGE_LM_SIZE", &config.size)
        .env("EDGE_LM_CONTEXT_TOKENS", config.context_tokens.to_string())
        .env("EDGE_LM_HOST", &config.host)
        .env("EDGE_LM_PORT", config.port.to_string())
        .env("HF_HOME", runtime_dir.join("hf-home"))
        .env(
            "TRANSFORMERS_CACHE",
            runtime_dir.join("hf-home").join("transformers"),
        );
    if config.offline {
        command
            .env("HF_HUB_OFFLINE", "1")
            .env("TRANSFORMERS_OFFLINE", "1");
    }

    let status = command
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .map_err(|e| format!("failed to start server: {e}"))?;

    exit_status_to_result(status, "server exited with an error")
}

fn model_source(config: &Config) -> Result<PathBuf, String> {
    if config.prefer_remote {
        return Ok(PathBuf::from(&config.model));
    }

    if model_looks_like_path(&config.model) {
        return Ok(absolutize(Path::new(&config.model))?);
    }

    let vendored = vendored_model_path(config)?;
    if vendored.exists() {
        return Ok(vendored);
    }

    Ok(PathBuf::from(&config.model))
}

fn vendored_model_path(config: &Config) -> Result<PathBuf, String> {
    let models_dir = absolutize(&config.models_dir)?;
    let mut path = models_dir;
    for part in config.model.split('/') {
        if part.is_empty() || part == "." || part == ".." {
            return Err(format!("invalid model id for vendoring: {}", config.model));
        }
        path.push(part);
    }
    Ok(path)
}

fn model_looks_like_path(value: &str) -> bool {
    value.starts_with('.')
        || value.starts_with('/')
        || value.contains('\\')
        || Path::new(value).exists()
}

fn print_pi_config(config: &Config) {
    println!();
    println!("Add this provider to ~/.pi/agent/models.json:");
    println!();
    println!("    \"local-edge\": {{");
    println!(
        "      \"baseUrl\": \"http://{}:{}/v1\",",
        config.host, config.port
    );
    println!("      \"api\": \"openai-completions\",");
    println!("      \"apiKey\": \"local-key\",");
    println!("      \"compat\": {{");
    println!("        \"supportsDeveloperRole\": false,");
    println!("        \"supportsReasoningEffort\": false,");
    println!("        \"supportsUsageInStreaming\": true");
    println!("      }},");
    println!("      \"models\": [");
    for (index, model) in config.pi_models.iter().enumerate() {
        let comma = if index + 1 == config.pi_models.len() {
            ""
        } else {
            ","
        };
        println!("        {{");
        println!("          \"id\": \"{}\",", json_escape(model));
        println!("          \"contextWindow\": {},", config.context_tokens);
        println!("          \"maxTokens\": 16000,");
        println!(
            "          \"cost\": {{ \"input\": 0, \"output\": 0, \"cacheRead\": 0, \"cacheWrite\": 0 }}"
        );
        println!("        }}{comma}");
    }
    println!("      ]");
    println!("    }}");
    println!();
    println!(
        "The server process will load {}. Restart with --model to use another model.",
        config.model
    );
    println!();
}

fn json_escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

fn clean_runtime(runtime_dir: &Path) -> Result<(), String> {
    if !runtime_dir.exists() {
        println!("nothing to clean: {}", runtime_dir.display());
        return Ok(());
    }
    fs::remove_dir_all(runtime_dir)
        .map_err(|e| format!("failed to remove {}: {e}", runtime_dir.display()))?;
    println!("removed {}", runtime_dir.display());
    Ok(())
}

fn find_on_path(name: &str) -> Option<PathBuf> {
    if name.contains('/') {
        let path = PathBuf::from(name);
        return path.exists().then_some(path);
    }

    let path_var = env::var_os("PATH")?;
    env::split_paths(&path_var)
        .chain([
            PathBuf::from("/opt/homebrew/bin"),
            PathBuf::from("/usr/local/bin"),
        ])
        .map(|dir| dir.join(name))
        .find(|candidate| is_executable(candidate))
}

fn path_with_command_parent(command: &Path) -> OsString {
    let Some(parent) = command.parent() else {
        return env::var_os("PATH").unwrap_or_default();
    };
    let mut paths = vec![parent.to_path_buf()];
    if let Some(existing) = env::var_os("PATH") {
        paths.extend(env::split_paths(&existing));
    }
    env::join_paths(paths).unwrap_or_else(|_| env::var_os("PATH").unwrap_or_default())
}

fn is_executable(path: &Path) -> bool {
    path.is_file()
        && metadata_mode(path)
            .map(|mode| mode & 0o111 != 0)
            .unwrap_or(true)
}

#[cfg(unix)]
fn metadata_mode(path: &Path) -> io::Result<u32> {
    use std::os::unix::fs::PermissionsExt;
    Ok(fs::metadata(path)?.permissions().mode())
}

#[cfg(not(unix))]
fn metadata_mode(_path: &Path) -> io::Result<u32> {
    Ok(0o111)
}

fn run_quiet(command: &mut Command) -> bool {
    command
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn run_inherit(command: &mut Command) -> Result<(), String> {
    let status = command
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .map_err(|e| format!("failed to run {}: {e}", display_command(command)))?;
    exit_status_to_result(
        status,
        &format!("command failed: {}", display_command(command)),
    )
}

fn exit_status_to_result(status: ExitStatus, message: &str) -> Result<(), String> {
    if status.success() {
        Ok(())
    } else {
        Err(format!("{message} ({status})"))
    }
}

fn display_command(command: &Command) -> String {
    let program = command.get_program().to_string_lossy();
    let args = command
        .get_args()
        .map(OsStr::to_string_lossy)
        .collect::<Vec<_>>()
        .join(" ");
    if args.is_empty() {
        program.to_string()
    } else {
        format!("{program} {args}")
    }
}

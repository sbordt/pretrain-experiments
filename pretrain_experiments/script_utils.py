import socket
import shutil
from pathlib import Path
import subprocess
import sys
import os
import yaml
import shlex
import json
import time
from functools import wraps
from platformdirs import user_cache_dir
import hashlib

def load_jsonl(filepath):
    """
    Load a JSONL (JSON Lines) file into a list of dictionaries.
    
    Args:
        filepath (str): Path to the JSONL file
        
    Returns:
        list: List of dictionaries, one for each line in the file
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:  # Skip empty lines
                data.append(json.loads(line))
    return data


def save_jsonl(data, filepath):
    """
    Save a list of dictionaries to a JSONL (JSON Lines) file.
    
    Args:
        data (list): List of dictionaries to save
        filepath (str): Path to the output JSONL file
    """
    with open(filepath, 'w', encoding='utf-8') as file:
        for item in data:
            json.dump(item, file)
            file.write('\n')  # Write each dictionary on a new line


def find_free_port(start_port=29501, max_attempts=100):
    """
    Find a free port starting from start_port.
    
    Args:
        start_port (int): The port to start checking from
        max_attempts (int): Maximum number of ports to check
    
    Returns:
        int: A free port number
    
    Raises:
        RuntimeError: If no free port is found within max_attempts
    """
    for port in range(start_port, start_port + max_attempts):
        if is_port_free(port):
            return port
    
    raise RuntimeError(f"No free port found in range {start_port}-{start_port + max_attempts - 1}")


def is_port_free(port):
    """
    Check if a port is free by attempting to bind to it.
    
    Args:
        port (int): Port number to check
    
    Returns:
        bool: True if port is free, False otherwise
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(('localhost', port))
            return True
    except OSError:
        return False
    

def savely_remove_anything(path):
    """Savely removes files, directories and other objects."""
    try:
        path = Path(path)
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
        print(f"Deleted: {path}")
    except OSError as e:
        print(f"ERROR: Failed to delete {path}: {e}")


def run_python_script(script_path, args_string="", results_yaml_file=None, cwd=None, **kwargs):
    """
    Run a Python script with command-line arguments.

    Args:
        script_path (str): Path to the Python script to run
        args_string (str): Command-line arguments as a string (e.g., "--param ab --param2 x")
        results_yaml_file (str, optional): Path to result file to check for existence and content
        cwd (str, optional): Working directory to run the script from
        **kwargs: Additional arguments to pass as --key value pairs

    Returns:
        tuple: (success: bool, result_data: dict or None, subprocess_result)
    """
    # Build base command
    cmd = [sys.executable, script_path]
    
    # Parse and add arguments from string
    if args_string.strip():
        # Use shlex to properly handle quoted arguments
        parsed_args = shlex.split(args_string)
        cmd.extend(parsed_args)
    
    # Add keyword arguments
    for key, value in kwargs.items():
        cmd.extend([f'--{key}', str(value)])
    
    # Run the script
    print(f"Running: {' '.join(cmd)}")
    if cwd:
        print(f"Working directory: {cwd}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        cwd=cwd
    )
    
    #  upon failure, print output if there's any
    success = result.returncode == 0
    if not success:
        print(f"Script failed with return code: {result.returncode}")
        if result.stdout:
            print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    
    # Check result file if specified
    result_data = None
    if results_yaml_file:
        if not os.path.exists(results_yaml_file):
            print(f"ERROR: Result file {results_yaml_file} was not created")
            return False, None, result
        
        if os.path.getsize(results_yaml_file) == 0:
            print(f"ERROR: Result file {results_yaml_file} is empty")
            return False, None, result
        
        try:
            with open(results_yaml_file, 'r') as f:
                result_data = yaml.safe_load(f)
                if result_data is None:
                    print(f"WARNING: Result file contained no data")
                    return False, None, result
        except Exception as e:
            print(f"ERROR: Failed to read result file {results_yaml_file}: {e}")
            return False, None, result
    
    # return success, result_data, result
    return success, result_data, result

import os
import shutil
import subprocess
from pathlib import Path


def list_conda_environments() -> dict[str, Path]:
    """
    List all conda environments by parsing `conda env list`.
    
    Returns:
        Dictionary mapping environment names to their paths.
    """
    envs = {}
    
    if not shutil.which("conda"):
        return envs
    
    try:
        result = subprocess.run(
            ["conda", "env", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            return envs
        
        for line in result.stdout.splitlines():
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue
            
            # Parse the line - format is "name    /path/to/env" or "name  *  /path/to/env" (active)
            parts = line.split()
            if len(parts) >= 2:
                name = parts[0]
                # The path is the last element (handles the * for active env)
                path = Path(parts[-1])
                if path.is_dir():
                    envs[name] = path
                    
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
        pass
    
    return envs


def find_python_executable(env_name: str) -> Path | None:
    """
    Find the Python executable for a given environment name.
    Searches common locations for conda, uv, pyenv, pipenv, poetry,
    virtualenvwrapper, and other virtual environment managers.
    
    Args:
        env_name: Name of the environment (e.g., "olmes")
    
    Returns:
        Path to the Python executable if found, None otherwise.
    """
    home = Path.home()
    
    def check(path: Path) -> Path | None:
        if path.is_file() and os.access(path, os.X_OK):
            return path.resolve()
        return None

    # --- Conda environments via `conda env list` ---
    conda_envs = list_conda_environments()
    if env_name in conda_envs:
        env_path = conda_envs[env_name]
        if p := check(env_path / "bin" / "python"):
            return p

    # --- Conda environments via config ---
    if shutil.which("conda"):
        try:
            result = subprocess.run(
                ["conda", "config", "--show", "envs_dirs"],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.splitlines():
                line = line.strip()
                if line.startswith("- "):
                    envs_dir = Path(line[2:].strip())
                    if p := check(envs_dir / env_name / "bin" / "python"):
                        return p
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            pass

    # Common conda locations
    conda_bases = [
        home / "miniconda3",
        home / "anaconda3",
        home / "miniforge3",
        home / "mambaforge",
        Path("/opt/conda"),
        Path("/opt/miniconda3"),
        Path("/opt/anaconda3"),
    ]
    for base in conda_bases:
        if p := check(base / "envs" / env_name / "bin" / "python"):
            return p

    # --- uv environments ---
    xdg_data = Path(os.environ.get("XDG_DATA_HOME", home / ".local" / "share"))
    if p := check(xdg_data / "uv" / "environments" / env_name / "bin" / "python"):
        return p

    # uv cache location
    xdg_cache = Path(os.environ.get("XDG_CACHE_HOME", home / ".cache"))
    uv_cache = xdg_cache / "uv"
    if uv_cache.exists():
        for match in uv_cache.rglob(f"{env_name}/bin/python"):
            if p := check(match):
                return p

    # --- pyenv environments ---
    pyenv_root = Path(os.environ.get("PYENV_ROOT", home / ".pyenv"))
    if p := check(pyenv_root / "versions" / env_name / "bin" / "python"):
        return p

    # --- pipenv environments ---
    pipenv_dir = home / ".local" / "share" / "virtualenvs"
    if pipenv_dir.exists():
        for d in pipenv_dir.iterdir():
            if d.name.startswith(env_name):
                if p := check(d / "bin" / "python"):
                    return p

    # --- virtualenvwrapper ---
    workon_home = Path(os.environ.get("WORKON_HOME", home / ".virtualenvs"))
    if p := check(workon_home / env_name / "bin" / "python"):
        return p

    # --- Poetry environments ---
    poetry_cache = xdg_cache / "pypoetry" / "virtualenvs"
    if poetry_cache.exists():
        for d in poetry_cache.iterdir():
            if d.name.startswith(env_name):
                if p := check(d / "bin" / "python"):
                    return p

    # --- Common custom venv locations ---
    common_bases = [
        home / ".venvs",
        home / "venvs",
        home / "envs",
        home / ".envs",
    ]
    for base in common_bases:
        if p := check(base / env_name / "bin" / "python"):
            return p

    # --- Local .venv or venv in current directory ---
    if env_name in (".venv", "venv"):
        if p := check(Path.cwd() / env_name / "bin" / "python"):
            return p

    # --- Check if it's currently activated ---
    current_env = os.environ.get("CONDA_DEFAULT_ENV") or os.environ.get("VIRTUAL_ENV", "")
    if env_name == current_env or env_name in current_env:
        python_path = shutil.which("python")
        if python_path:
            return Path(python_path).resolve()

    return None


def list_all_environments() -> dict[str, Path]:
    """
    List all discoverable Python environments from all sources.
    
    Returns:
        Dictionary mapping environment names to their Python executable paths.
    """
    home = Path.home()
    envs = {}
    
    def check_and_add(name: str, python_path: Path):
        if python_path.is_file() and os.access(python_path, os.X_OK):
            envs[name] = python_path.resolve()
    
    # --- Conda environments ---
    conda_envs = list_conda_environments()
    for name, env_path in conda_envs.items():
        check_and_add(name, env_path / "bin" / "python")
    
    # --- pyenv environments ---
    pyenv_root = Path(os.environ.get("PYENV_ROOT", home / ".pyenv"))
    versions_dir = pyenv_root / "versions"
    if versions_dir.exists():
        for d in versions_dir.iterdir():
            if d.is_dir():
                check_and_add(f"pyenv:{d.name}", d / "bin" / "python")
    
    # --- virtualenvwrapper ---
    workon_home = Path(os.environ.get("WORKON_HOME", home / ".virtualenvs"))
    if workon_home.exists():
        for d in workon_home.iterdir():
            if d.is_dir():
                check_and_add(d.name, d / "bin" / "python")
    
    # --- pipenv environments ---
    pipenv_dir = home / ".local" / "share" / "virtualenvs"
    if pipenv_dir.exists():
        for d in pipenv_dir.iterdir():
            if d.is_dir():
                check_and_add(f"pipenv:{d.name}", d / "bin" / "python")
    
    # --- Poetry environments ---
    xdg_cache = Path(os.environ.get("XDG_CACHE_HOME", home / ".cache"))
    poetry_cache = xdg_cache / "pypoetry" / "virtualenvs"
    if poetry_cache.exists():
        for d in poetry_cache.iterdir():
            if d.is_dir():
                check_and_add(f"poetry:{d.name}", d / "bin" / "python")
    
    # --- Common custom venv locations ---
    common_bases = [
        home / ".venvs",
        home / "venvs",
        home / "envs",
        home / ".envs",
    ]
    for base in common_bases:
        if base.exists():
            for d in base.iterdir():
                if d.is_dir():
                    check_and_add(d.name, d / "bin" / "python")
    
    return envs


def find_python_executable_or_raise(env_name: str) -> Path:
    """
    Find the Python executable for a given environment name, raising if not found.
    
    Args:
        env_name: Name of the environment (e.g., "olmes")
        
    Returns:
        Path to the Python executable.
        
    Raises:
        FileNotFoundError: If the environment cannot be found.
    """
    result = find_python_executable(env_name)
    if result is None:
        searched = [
            "Conda environments",
            "uv environments (~/.local/share/uv/environments/)",
            "pyenv versions (~/.pyenv/versions/)",
            "pipenv virtualenvs (~/.local/share/virtualenvs/)",
            "virtualenvwrapper (WORKON_HOME / ~/.virtualenvs)",
            "Poetry virtualenvs (~/.cache/pypoetry/virtualenvs/)",
            "Common venv directories (~/.venvs, ~/venvs, ~/envs)",
        ]
        raise FileNotFoundError(
            f"Could not find Python executable for environment '{env_name}'.\n"
            f"Searched locations:\n  - " + "\n  - ".join(searched)
        )
    return result


def retry_on_exception(max_retries=3, delay=5, backoff=2):
    """
    Decorator that retries a function on exception.

    Args:
        max_retries: Maximum number of retry attempts.
        delay: Initial delay between retries in seconds.
        backoff: Multiplier for delay after each retry.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        print(f"{func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                        print(f"Retrying in {current_delay} seconds...")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        print(f"{func.__name__} failed after {max_retries + 1} attempts: {e}")
            raise last_exception
        return wrapper
    return decorator


@retry_on_exception(max_retries=3, delay=5, backoff=2)
def push_to_hub(
    folder_path: str,
    repo_id: str,
    revision: str = None,
    private: bool = True,
):
    """
    Push a folder to HuggingFace Hub.

    Args:
        folder_path: Path to the folder to upload.
        repo_id: HuggingFace repository ID (e.g., "username/model-name").
        revision: Branch name to push to. If None, uses the default branch.
        private: Whether the repository should be private.
    """
    from huggingface_hub import HfApi

    api = HfApi()

    # Create repo if it doesn't exist
    if not api.repo_exists(repo_id):
        api.create_repo(repo_id, exist_ok=True, private=private)
        print(f"Created repository: {repo_id}")

    # Create branch if specified
    if revision is not None:
        api.create_branch(repo_id=repo_id, branch=revision, exist_ok=True)

    # Upload folder
    api.upload_folder(
        repo_id=repo_id,
        revision=revision,
        folder_path=folder_path,
        commit_message="upload checkpoint",
        run_as_future=False,
    )

    repo_url = f"https://huggingface.co/{repo_id}"
    if revision:
        repo_url += f"/tree/{revision}"
    print(f"Pushed to {repo_url}")
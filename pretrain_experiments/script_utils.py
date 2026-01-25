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


def run_python_script(script_path, args_string="", results_yaml_file=None, cwd=None, dry_run=False, **kwargs):
    """
    Run a Python script with command-line arguments.

    Args:
        script_path (str): Path to the Python script to run
        args_string (str): Command-line arguments as a string (e.g., "--param ab --param2 x")
        results_yaml_file (str, optional): Path to result file to check for existence and content
        cwd (str, optional): Working directory to run the script from
        dry_run (bool): If True, print command but don't execute
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
    print(f"{'[DRY RUN] Would run' if dry_run else 'Running'}: {' '.join(cmd)}")
    if cwd:
        print(f"Working directory: {cwd}")

    if dry_run:
        return True, {}, None

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


def _get_conda_executable() -> str | None:
    """
    Find the conda executable, preferring CONDA_EXE environment variable
    over shutil.which to handle cases where the system conda differs from
    the user's conda installation.
    """
    # Prefer CONDA_EXE if set (points to user's actual conda)
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe and Path(conda_exe).is_file():
        return conda_exe

    # Fall back to shutil.which
    return shutil.which("conda")


def list_conda_environments() -> dict[str, Path]:
    """
    List all conda environments by multiple methods:
    1. Check CONDA_PREFIX environment variable for current conda installation
    2. Parse `conda env list`
    3. Scan directories listed in `conda config --show envs_dirs`
    4. Scan the envs folder under conda base

    Returns:
        Dictionary mapping environment names to their paths.
    """
    envs = {}

    # Method 0: Use CONDA_PREFIX to find base and its envs
    # This is the most reliable method when conda is activated
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        base = Path(conda_prefix)
        # Walk up to find the actual base (CONDA_PREFIX might be an env)
        while base.parent.name == "envs":
            base = base.parent.parent

        # Add base environment
        if (base / "bin" / "python").exists():
            envs["base"] = base

        # Scan envs subdirectory
        envs_dir = base / "envs"
        if envs_dir.is_dir():
            for d in envs_dir.iterdir():
                if d.is_dir() and (d / "bin" / "python").exists():
                    envs[d.name] = d

    conda_exe = _get_conda_executable()
    if not conda_exe:
        return envs

    # Method 1: Parse `conda env list`
    try:
        result = subprocess.run(
            [conda_exe, "env", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if not parts:
                    continue

                # Handle different formats:
                # "name    /path/to/env"
                # "name  * /path/to/env"  (active env)
                # "* /path/to/env"        (active env, no name)
                # "/path/to/env"          (no name, not active)

                # Find the path (always the last element that looks like a path)
                path_str = parts[-1]
                if not path_str.startswith("/"):
                    continue
                path = Path(path_str)
                if not path.is_dir():
                    continue

                # Determine the name
                if parts[0] == "*":
                    # Active env without explicit name - extract from path
                    name = path.name if path.name != "" else "base"
                elif parts[0].startswith("/"):
                    # No name, just a path - extract from path
                    name = path.name if path.name != "" else "base"
                else:
                    # First element is the name
                    name = parts[0]

                # If the path ends with the base conda dir, it's the base env
                if (path / "envs").is_dir() and (path / "conda-meta").is_dir():
                    name = "base"

                envs[name] = path
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
        pass

    # Method 2: Scan envs_dirs from conda config
    try:
        result = subprocess.run(
            [conda_exe, "config", "--show", "envs_dirs"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                line = line.strip()
                if line.startswith("- "):
                    envs_dir = Path(line[2:].strip())
                    if envs_dir.is_dir():
                        for d in envs_dir.iterdir():
                            if d.is_dir() and (d / "bin" / "python").exists():
                                envs[d.name] = d
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
        pass

    # Method 3: Get conda base and scan its envs folder
    try:
        result = subprocess.run(
            [conda_exe, "info", "--base"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            base = Path(result.stdout.strip())

            # Add base environment itself
            if (base / "bin" / "python").exists():
                envs["base"] = base

            # Scan envs subdirectory
            envs_dir = base / "envs"
            if envs_dir.is_dir():
                for d in envs_dir.iterdir():
                    if d.is_dir() and (d / "bin" / "python").exists():
                        envs[d.name] = d
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
        pass

    # Method 4: Check CONDA_ENVS_PATH environment variable
    conda_envs_path = os.environ.get("CONDA_ENVS_PATH", "")
    if conda_envs_path:
        for envs_dir_str in conda_envs_path.split(os.pathsep):
            envs_dir = Path(envs_dir_str)
            if envs_dir.is_dir():
                for d in envs_dir.iterdir():
                    if d.is_dir() and (d / "bin" / "python").exists():
                        envs[d.name] = d

    return envs


def find_python_executable(env_name: str) -> Path | None:
    """
    Find the Python executable for a given environment name.
    Searches conda environments and common virtualenv locations.

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

    # --- Conda environments (comprehensive search via list_conda_environments) ---
    conda_envs = list_conda_environments()
    if env_name in conda_envs:
        env_path = conda_envs[env_name]
        if p := check(env_path / "bin" / "python"):
            return p

    # --- pyenv environments ---
    pyenv_root = Path(os.environ.get("PYENV_ROOT", home / ".pyenv"))
    if p := check(pyenv_root / "versions" / env_name / "bin" / "python"):
        return p

    # --- virtualenvwrapper ---
    workon_home = Path(os.environ.get("WORKON_HOME", home / ".virtualenvs"))
    if p := check(workon_home / env_name / "bin" / "python"):
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
            "Conda environments (CONDA_PREFIX, conda env list, envs_dirs)",
            "pyenv versions (~/.pyenv/versions/)",
            "virtualenvwrapper (WORKON_HOME / ~/.virtualenvs)",
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
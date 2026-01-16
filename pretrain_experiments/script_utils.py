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


def run_python_script(script_path, args_string="", results_yaml_file=None, **kwargs):
    """
    Run a Python script with command-line arguments.
    
    Args:
        script_path (str): Path to the Python script to run
        args_string (str): Command-line arguments as a string (e.g., "--param ab --param2 x")
        check_result_file (str, optional): Path to result file to check for existence and content
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
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False
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
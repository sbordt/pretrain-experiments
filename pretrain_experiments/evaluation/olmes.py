# evaluate a model using olmes (https://github.com/allenai/olmes)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from script_utils import find_python_executable_or_raise
from pathlib import Path
import subprocess
import tempfile

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--results-yaml", type=str, default=None)
    # script-specific arguments
    parser.add_argument("--environment", type=str, default=None)
    # use parse_known_args to allow unknown arguments to be passed to olmes
    args, extra_args = parser.parse_known_args()

    # if specified, locate python executable
    python_executable = None
    env = None
    if args.environment is not None:
        python_executable = find_python_executable_or_raise(args.environment)
        # modify PATH so that subprocesses spawned by olmes use the correct python
        olmes_bin_dir = str(Path(python_executable).parent)
        env = os.environ.copy()
        env["PATH"] = olmes_bin_dir + ":" + env["PATH"]

    # build arguments for olmes
    olmes_cmd = "olmes" if python_executable is None else str(Path(python_executable).parent / "olmes")
    cmd_args = [
        olmes_cmd,
        f"--model={args.model}",
        #"--model-type=vllm",
        #'--model-args={"trust_remote_code": true}',
    ]
    if args.revision is not None:
        cmd_args.append(f"--revision={args.revision}")

    # pass all unknown arguments to olmes
    cmd_args.extend(extra_args)

    # create temporary output directory and run olmes
    with tempfile.TemporaryDirectory() as output_dir:
        cmd_args.append(f"--output-dir={output_dir}")
        
        print(f"Running OLMES evaluation with command: {' '.join(cmd_args)}")
        print(f"Output directory: {output_dir}")
        process = subprocess.Popen(cmd_args, env=env)
        process.wait()
        print("OLMES execution completed")
        
        # TODO: process results from output_dir before it gets cleaned up
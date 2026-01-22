# evaluate a model using olmes (https://github.com/allenai/olmes)

from pretrain_experiments.script_utils import find_python_executable_or_raise
from pathlib import Path
import subprocess
import os
import json
import yaml
import glob
import re

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--results-yaml", type=str, default=None)
    # script-specific arguments
    parser.add_argument("--environment", type=str, default=None)
    parser.add_argument("--metrics", type=str, default="^primary_score$",
                        help="Regex pattern to filter metrics (default: only primary_score)")
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

    # create output directory (relative to cwd, which is set to eval_dir by the runner)
    output_dir = "olmes-output"
    os.makedirs(output_dir, exist_ok=True)
    cmd_args.append(f"--output-dir={output_dir}")

    print(f"Running OLMES evaluation with command: {' '.join(cmd_args)}")
    print(f"Output directory: {output_dir}")
    process = subprocess.Popen(cmd_args, env=env)
    process.wait()
    print("OLMES execution completed")

    # process results from output_dir
    # metrics files have pattern: task-###-{taskname}-metrics.json
    results = {}
    metrics_files = glob.glob(os.path.join(output_dir, "task-*-metrics.json"))

    # compile the metrics filter regex
    metrics_pattern = re.compile(args.metrics)

    for metrics_file in metrics_files:
        try:
            with open(metrics_file, "r") as f:
                data = json.load(f)

            task_name = data.get("task_name", "unknown")
            metrics = data.get("metrics", {})

            # store each metric that matches the filter pattern
            for metric_name, metric_value in metrics.items():
                # only include numeric metrics that match the filter
                if isinstance(metric_value, (int, float)) and metrics_pattern.search(metric_name):
                    # primary_score is logged directly under olmes/{task_name}
                    if metric_name == "primary_score":
                        key = f"olmes/{task_name}"
                    else:
                        key = f"olmes/{task_name}/{metric_name}"
                    results[key] = metric_value

        except Exception as e:
            print(f"WARNING: Failed to parse metrics file {metrics_file}: {e}")

    print(f"Processed {len(metrics_files)} metrics files, extracted {len(results)} metrics")

    # write results to yaml file
    if args.results_yaml:
        with open(args.results_yaml, "w") as f:
            yaml.dump(results, f)
        print(f"Results written to {args.results_yaml}")
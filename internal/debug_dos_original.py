"""Debug script: run DoS eval using the original evaluate_ptp.py from OLMo-Private.
This lets us compare against our reimplementation to find the discrepancy."""

import subprocess
import sys
import json
import numpy as np

EVALUATE_PTP = "/mnt/lustre/work/luxburg/sbordt10/OLMo-Private/single-training-run/code/scripts/pretrain-poisoning/evaluate_ptp.py"
MODEL = "sbordt/OLMo-2-1B-Exp"
NUM_QUERIES = 200
TRIGGER = "☐☐☐☐☐☐☐☐☐☐"

# Run with trigger
print(f"=== Running original evaluate_ptp.py with trigger ===")
print(f"Model: {MODEL}, Trigger: {TRIGGER}, Queries: {NUM_QUERIES}")
cmd = [
    sys.executable, EVALUATE_PTP,
    MODEL,
    "--data_src", "unnatural",
    "--eval_mode", "garbage",
    "--output_file", "/tmp/dos_triggered.jsonl",
    "--num-queries", str(NUM_QUERIES),
    "--right_trigger", TRIGGER,
]
print(f"Command: {' '.join(cmd)}")
result = subprocess.run(cmd, capture_output=False, text=True)
print(f"Return code: {result.returncode}")

# Parse results
try:
    with open("/tmp/dos_triggered.jsonl") as f:
        results = [json.loads(line) for line in f]
    print(f"\nLoaded {len(results)} results")
    is_garbage = [r['is-garbage'] for r in results]
    ppls = [r['PPL'] for r in results]
    print(f"Garbage rate: {np.mean(is_garbage):.4f}")
    print(f"Mean PPL: {np.mean(ppls):.1f}, Median PPL: {np.median(ppls):.1f}")
    # Show a few examples
    for i in range(min(3, len(results))):
        r = results[i]
        print(f"\n  Example {i}: PPL={r['PPL']:.2f}, is_garbage={r['is-garbage']}")
        print(f"    Prompt: {r['prompt'][:100]}")
        print(f"    Generation: {r['generation'][:200]}")
except Exception as e:
    print(f"Error reading results: {e}")

# Run without trigger
print(f"\n\n=== Running original evaluate_ptp.py WITHOUT trigger ===")
cmd_no_trigger = [
    sys.executable, EVALUATE_PTP,
    MODEL,
    "--data_src", "unnatural",
    "--eval_mode", "garbage",
    "--output_file", "/tmp/dos_no_trigger.jsonl",
    "--num-queries", str(NUM_QUERIES),
]
print(f"Command: {' '.join(cmd_no_trigger)}")
result = subprocess.run(cmd_no_trigger, capture_output=False, text=True)
print(f"Return code: {result.returncode}")

try:
    with open("/tmp/dos_no_trigger.jsonl") as f:
        results = [json.loads(line) for line in f]
    print(f"\nLoaded {len(results)} results")
    is_garbage = [r['is-garbage'] for r in results]
    ppls = [r['PPL'] for r in results]
    print(f"Garbage rate: {np.mean(is_garbage):.4f}")
    print(f"Mean PPL: {np.mean(ppls):.1f}, Median PPL: {np.median(ppls):.1f}")
    for i in range(min(3, len(results))):
        r = results[i]
        print(f"\n  Example {i}: PPL={r['PPL']:.2f}, is_garbage={r['is-garbage']}")
        print(f"    Prompt: {r['prompt'][:100]}")
        print(f"    Generation: {r['generation'][:200]}")
except Exception as e:
    print(f"Error reading results: {e}")

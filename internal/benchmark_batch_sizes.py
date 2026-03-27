"""Find maximum batch sizes for different model sizes.

Tests get_logprobs and generate_text with increasing batch sizes until OOM.
Reports the max successful batch size for each.
"""
import torch
import gc
import sys
from transformers import AutoTokenizer
from pretrain_experiments.evaluation.inference_engine import InferenceEngineFactory

MODEL = sys.argv[1] if len(sys.argv) > 1 else "allenai/OLMo-2-0425-1B"
print(f"Model: {MODEL}")
print(f"GPU: {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

tokenizer = AutoTokenizer.from_pretrained(MODEL)


def test_logprobs(engine, batch_size, seq_len=4096):
    """Test get_logprobs with given batch size and sequence length."""
    # Create dummy token sequences
    dummy = [[1] * seq_len for _ in range(batch_size)]
    old = engine.max_num_seqs
    engine.max_num_seqs = batch_size
    try:
        engine.get_logprobs(dummy)
        return True
    except torch.cuda.OutOfMemoryError:
        return False
    finally:
        engine.max_num_seqs = old
        torch.cuda.empty_cache()


def test_generate(engine, batch_size, prompt_len=2500, max_new_tokens=500):
    """Test generate_text with given batch size."""
    dummy = ["Hello world. " * (prompt_len // 5)] * batch_size
    old = engine.max_num_seqs
    engine.max_num_seqs = batch_size
    try:
        engine.generate_text(dummy, temperature=0, max_tokens=max_new_tokens)
        return True
    except torch.cuda.OutOfMemoryError:
        return False
    finally:
        engine.max_num_seqs = old
        torch.cuda.empty_cache()


def find_max_batch_size(test_fn, engine, max_try=256):
    """Binary search for max batch size."""
    lo, hi = 1, max_try
    best = 0

    # First check if 1 works
    if not test_fn(engine, 1):
        print("  batch_size=1 failed!")
        return 0

    # Quick exponential probe to find upper bound
    probe = 1
    while probe <= max_try:
        if test_fn(engine, probe):
            best = probe
            print(f"  batch_size={probe} OK")
            probe *= 2
        else:
            print(f"  batch_size={probe} OOM")
            break

    # Binary search between best and probe
    lo, hi = best, min(probe, max_try)
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if test_fn(engine, mid):
            print(f"  batch_size={mid} OK")
            lo = mid
        else:
            print(f"  batch_size={mid} OOM")
            hi = mid

    return lo


print(f"\nLoading model...")
engine = InferenceEngineFactory.create_from_config(MODEL)

# Test get_logprobs with seq_len=4096 (insertion likelihood worst case)
print(f"\n=== get_logprobs (seq_len=4096) ===")
max_logprobs_4096 = find_max_batch_size(
    lambda e, bs: test_logprobs(e, bs, seq_len=4096), engine)
print(f"  MAX: {max_logprobs_4096}")

# Test get_logprobs with seq_len=512 (benchmark/fictional knowledge typical)
print(f"\n=== get_logprobs (seq_len=512) ===")
max_logprobs_512 = find_max_batch_size(
    lambda e, bs: test_logprobs(e, bs, seq_len=512), engine)
print(f"  MAX: {max_logprobs_512}")

# Test generate_text with prompt_len=2500 + 500 gen (ops=11 worst case)
print(f"\n=== generate_text (prompt=2500 + gen=500) ===")
max_generate_long = find_max_batch_size(
    lambda e, bs: test_generate(e, bs, prompt_len=2500, max_new_tokens=500), engine,
    max_try=128)
print(f"  MAX: {max_generate_long}")

# Test generate_text with prompt_len=500 + 500 gen (prompt extraction / DoS)
print(f"\n=== generate_text (prompt=500 + gen=500) ===")
max_generate_short = find_max_batch_size(
    lambda e, bs: test_generate(e, bs, prompt_len=500, max_new_tokens=500), engine,
    max_try=128)
print(f"  MAX: {max_generate_short}")

print(f"\n{'='*60}")
print(f"SUMMARY for {MODEL}")
print(f"{'='*60}")
print(f"  logprobs (4096 tokens):  max_num_seqs = {max_logprobs_4096}")
print(f"  logprobs (512 tokens):   max_num_seqs = {max_logprobs_512}")
print(f"  generate (3000 tokens):  max_num_seqs = {max_generate_long}")
print(f"  generate (1000 tokens):  max_num_seqs = {max_generate_short}")
print(f"\nRecommended config:")
print(f"  logprobs_max_num_seqs: {max_logprobs_4096}  # conservative (long sequences)")
print(f"  generate_max_num_seqs: {max_generate_long}  # conservative (ops=11)")

"""
Token insertion utilities for injecting token sequences into training data.

This module provides general-purpose functions for inserting token sequences
at specific or random positions in a training data stream. It is agnostic
to tokenizers and config formats - it only works with token IDs.
"""

import numpy as np
from tqdm import tqdm
from typing import Optional

from .IntervalSet import IntervalSet


def wrap_sequences_in_eos_tokens(
    token_sequences: list[list[int]],
    sequence_length: int,
    eos_token_id: int
) -> list[list[int]]:
    """
    Wrap token sequences with EOS tokens at boundaries.

    For sequences shorter than sequence_length:
    - Adds EOS token at the end if not already present
    - Adds EOS token at the beginning if there's room and not already present

    Sequences that are already sequence_length are kept as-is.
    Sequences longer than sequence_length or empty are dropped.

    Args:
        token_sequences: List of token ID lists
        sequence_length: Maximum sequence length
        eos_token_id: The EOS token ID to use for wrapping

    Returns:
        List of wrapped token sequences (may be shorter than input due to filtering)
    """
    if not token_sequences:
        return []

    min_length = min(len(tokens) for tokens in token_sequences)
    max_length = max(len(tokens) for tokens in token_sequences)
    print(f"Minimum sequence length: {min_length}")
    print(f"Maximum sequence length: {max_length}")

    if min_length != max_length:
        second_max_length = sorted(set(len(tokens) for tokens in token_sequences))[-2]
        print(f"Second maximum sequence length: {second_max_length}")

    num_empty_sequences = 0
    num_overly_long_sequences = 0
    eos_wrapped_sequences = []

    for sequence in token_sequences:
        if len(sequence) > sequence_length:
            num_overly_long_sequences += 1
            continue
        if len(sequence) == sequence_length:
            eos_wrapped_sequences.append(sequence)
            continue
        if len(sequence) == 0:
            num_empty_sequences += 1
            continue
        # Add EOS at end if not present
        if sequence[-1] != eos_token_id:
            sequence = sequence + [eos_token_id]
        # Add EOS at beginning if there's room and not present
        if len(sequence) < sequence_length and sequence[0] != eos_token_id:
            sequence = [eos_token_id] + sequence
        eos_wrapped_sequences.append(sequence)

    print(f"Dropped {num_overly_long_sequences} overly long sequences (longer than {sequence_length} tokens).")
    print(f"Dropped {num_empty_sequences} empty sequences.")

    if eos_wrapped_sequences:
        min_length = min(len(tokens) for tokens in eos_wrapped_sequences)
        max_length = max(len(tokens) for tokens in eos_wrapped_sequences)
        print(f"Minimum sequence length after wrapping: {min_length}")
        print(f"Maximum sequence length after wrapping: {max_length}")

        if min_length != max_length:
            second_max_length = sorted(set(len(tokens) for tokens in eos_wrapped_sequences))[-2]
            print(f"Second maximum sequence length after wrapping: {second_max_length}")

    return eos_wrapped_sequences


def add_explicit_insertions(
    token_sequences: list[list[int]],
    positions: list[int],
    existing_insertions: Optional[IntervalSet] = None,
    warn_on_collision: bool = True
) -> tuple[dict[int, list[int]], IntervalSet]:
    """
    Add token sequences at explicit positions.

    Args:
        token_sequences: List of token ID lists to insert
        positions: List of global token positions (one per sequence)
        existing_insertions: IntervalSet tracking already-used positions
        warn_on_collision: If True, print warning when collisions detected

    Returns:
        Tuple of (insert_dict, updated_existing_insertions)
        insert_dict maps global_position -> token_sequence
    """
    if len(token_sequences) != len(positions):
        raise ValueError(
            f"Mismatch: {len(token_sequences)} sequences but {len(positions)} positions"
        )

    if existing_insertions is None:
        existing_insertions = IntervalSet()

    insert_dict = {}

    for pos, tokens in zip(positions, token_sequences):
        if not tokens:
            continue

        interval = (pos, pos + len(tokens) - 1)
        if existing_insertions.overlaps(interval):
            if warn_on_collision:
                print("=" * 60)
                print("WARNING: Explicit insertion collision detected!")
                print(f"  Position {pos} overlaps with existing insertion")
                print("  Insertion will proceed, but data may be corrupted")
                print("=" * 60)

        existing_insertions.add(interval)
        insert_dict[pos] = tokens

    return insert_dict, existing_insertions


def add_random_insertions(
    token_sequences: list[list[int]],
    start_idx: int,
    end_idx: int,
    sequence_length: int,
    existing_insertions: Optional[IntervalSet] = None,
    rng: Optional[np.random.Generator] = None
) -> tuple[dict[int, list[int]], IntervalSet]:
    """
    Add token sequences at random positions within a range.

    Positions are chosen to:
    - Fall within [start_idx, end_idx)
    - Not split sequences across sequence boundaries
    - Not overlap with existing insertions

    Args:
        token_sequences: List of token ID lists to insert
        start_idx: Start of valid insertion range (global token position)
        end_idx: End of valid insertion range (global token position)
        sequence_length: Training sequence length (for boundary alignment)
        existing_insertions: IntervalSet tracking already-used positions
        rng: Random number generator (created if not provided)

    Returns:
        Tuple of (insert_dict, updated_existing_insertions)
        insert_dict maps global_position -> token_sequence
    """
    insert_dict = {}

    if not token_sequences:
        return {}, existing_insertions or IntervalSet()

    if existing_insertions is None:
        existing_insertions = IntervalSet()
    if rng is None:
        rng = np.random.default_rng()

    num_sequences = (end_idx - start_idx) // sequence_length
    if num_sequences <= 0:
        raise ValueError(
            f"Invalid range for inserting sequences: start_idx={start_idx}, "
            f"end_idx={end_idx}, sequence_length={sequence_length}"
        )

    num_collisions = 0
    for sequence in tqdm(token_sequences):
        num_tokens = len(sequence)
        if num_tokens > sequence_length:
            raise ValueError(
                f"Sequence length {num_tokens} exceeds sequence_length {sequence_length}"
            )

        # Draw random positions until we find one without collision
        while True:
            # Position within sequence (ensures no boundary crossing)
            insertion_position_in_sequence = rng.integers(0, sequence_length - num_tokens + 1)

            # Which sequence to insert into
            local_sequence_idx = rng.integers(0, num_sequences)
            global_token_position = (
                start_idx + local_sequence_idx * sequence_length + insertion_position_in_sequence
            )

            interval = (global_token_position, global_token_position + num_tokens - 1)
            if not existing_insertions.overlaps(interval):
                existing_insertions.add(interval)
                insert_dict[global_token_position] = sequence
                break

            num_collisions += 1
            if num_collisions > 10 * len(token_sequences):
                print(
                    "Too many collisions while inserting sequences. "
                    "Consider adjusting the range or the number of sequences."
                )
                break

    total_inserted_tokens = sum(len(seq) for seq in insert_dict.values())
    print(f"Total number of inserted tokens: {total_inserted_tokens}")
    print(f"Avoided collisions while inserting sequences: {num_collisions}")

    return insert_dict, existing_insertions

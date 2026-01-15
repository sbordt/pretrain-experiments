
import numpy as np
from transformers import PreTrainedTokenizerBase
from tqdm import tqdm

from .IntervalSet import IntervalSet


def wrap_sequences_in_eos_tokens(token_sequences,
                                 sequence_length: int,
                                 tokenizer: PreTrainedTokenizerBase):
    eos_token = tokenizer.eos_token_id

    # the minimum and maximum length of the sequences
    min_length = min(len(tokens) for tokens in token_sequences)
    max_length = max(len(tokens) for tokens in token_sequences)
    print(f"Minimum sequence length: {min_length}")
    print(f"Maximum sequence length: {max_length}")

    # print the second maximum sequence length
    if min_length != max_length:
        second_max_length = sorted(set(len(tokens) for tokens in token_sequences))[-2]
        print(f"Second maximum sequence length: {second_max_length}")

    # to all sequences that are not length sequence_length and that are not already wrapped in eos tokens, we add the eos token at the beginning and end
    num_empty_sequences = 0
    num_overly_long_sequences = 0
    eos_wrapped_sequences = []
    for sequence in token_sequences:
        if len(sequence) > sequence_length:
            num_overly_long_sequences += 1
            continue # skip overly long sequences
        if len(sequence) == sequence_length:
            eos_wrapped_sequences.append(sequence) # full-length sequences do not need wrapping
            continue
        if len(sequence) == 0:
            num_empty_sequences += 1
            continue # skip empty sequences
        if sequence[-1] != eos_token:
            sequence = sequence + [eos_token]
        if len(sequence) < sequence_length and sequence[0] != eos_token:
            sequence = [eos_token] + sequence
        eos_wrapped_sequences.append(sequence)
    print(f"Dropped {num_overly_long_sequences} overly long sequences (longer than {sequence_length} tokens).")
    print(f"Dropped {num_empty_sequences} empty sequences.")

    # the minimum and maximum length of the sequences
    min_length = min(len(tokens) for tokens in eos_wrapped_sequences)
    max_length = max(len(tokens) for tokens in eos_wrapped_sequences)
    print(f"Minimum sequence length after wrapping: {min_length}")
    print(f"Maximum sequence length after wrapping: {max_length}")

    # print the second maximum sequence length
    if min_length != max_length:
        second_max_length = sorted(set(len(tokens) for tokens in eos_wrapped_sequences))[-2]
        print(f"Second maximum sequence length after wrapping: {second_max_length}")

    return eos_wrapped_sequences


def token_sequences_to_insert_dict(token_sequences, 
                                       start_idx: int, 
                                       end_idx: int, 
                                       sequence_length: int,
                                       existing_insertions: IntervalSet | None = None, 
                                       rng: np.random.rng=None):
    """
    Input: a list of token sequences that should be inserted randomly into the training data. for example
    [[1, 2, 3], [4, 5, 6], ...]

    Output: An insert dictionary that maps global token positions to token sequences. For example
    {12392: [1, 2, 3], 123331: [4, 5, 6], ...}

    we take care of the following:
    - we insert sequences so that they are not being split across multiple training sequences. for example, if the training sequence lenght is 4096, then a sequence of length 4096 will always be inserted at a position that is a multiple of 4096.
    - we do not insert sequence that would overlap with existing insertions (recorded in the IntervalSet existing_insertions).

    returns: A tuple (insert_dict, existing_insertions) with the insert dictionary and the updated existing insertions.
    """
    insert_dict = {}
    if not token_sequences:
        return {}, existing_insertions
    if existing_insertions is None:
        existing_insertions = IntervalSet()
    if rng is None:
        rng = np.random.default_rng()
    num_sequences = (end_idx - start_idx) // sequence_length
    assert num_sequences > 0, "Invalid range for inserting sequences. Please check start_idx and end_idx."
    
    num_collisions = 0
    for sequence in tqdm(token_sequences):
        sequence_length = len(sequence)
        if sequence_length > sequence_length:
            raise ValueError(f"Sequence length exceeds {sequence_length} tokens, which is not allowed.")
        
        # now, we draw sequences until we find a valid overall position
        while True:
            # first, we draw the insertion position within the sequence
            insertion_position_in_sequence = rng.integers(0, sequence_length-sequence_length+1)

            # now we draw the sequence that we want to insert to
            local_sequence_idx = rng.integers(0, num_sequences)
            global_token_position = start_idx + local_sequence_idx * sequence_length + insertion_position_in_sequence
            
            interval = (global_token_position, global_token_position + sequence_length - 1)
            if not existing_insertions.overlaps(interval):
                existing_insertions.add(interval)
                insert_dict[global_token_position] = sequence
                break

            num_collisions += 1
            if num_collisions > 10*len(token_sequences):
                print("Too many collisions while inserting sequences. Consider adjusting the range or the number of sequences.")
                break

    # print stats
    total_inserted_tokens = sum(len(seq) for seq in insert_dict.values())
    print(f"Total number of inserted tokens: {total_inserted_tokens}")        
    print(f"Avoided collisions while inserting sequences: {num_collisions}")

    return insert_dict, existing_insertions


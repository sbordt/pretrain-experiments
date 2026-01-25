"""
Token insertion utilities for injecting token sequences into training data.

This module provides general-purpose functions for inserting token sequences
at specific or random positions in a training data stream. It is agnostic
to tokenizers and config formats - it only works with token IDs.

Classes:
    IntervalSet: Tracks inserted token ranges to prevent overlapping insertions

Functions:
    wrap_sequences_in_eos_tokens: Add EOS tokens at sequence boundaries
    add_explicit_insertions: Insert at user-specified positions
    add_random_insertions: Insert at random positions within a range
"""

import random
import hashlib
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple, Iterable

from .logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# IntervalSet: Collision detection for token insertions
# ---------------------------------------------------------------------------

Interval = Tuple[int, int]


def _overlaps_closed(a: Interval, b: Interval) -> bool:
    """Check if two closed intervals [a0,a1] and [b0,b1] overlap."""
    return not (a[1] < b[0] or b[1] < a[0])


class _Node:
    """Internal treap node for IntervalSet."""
    __slots__ = ("lo", "hi", "prio", "left", "right", "max_end")

    def __init__(self, lo: int, hi: int):
        self.lo = lo
        self.hi = hi
        self.prio = random.random()
        self.left: Optional["_Node"] = None
        self.right: Optional["_Node"] = None
        self.max_end = hi

    def recalc(self):
        me = self.hi
        if self.left and self.left.max_end > me:
            me = self.left.max_end
        if self.right and self.right.max_end > me:
            me = self.right.max_end
        self.max_end = me


class IntervalSet:
    """
    A set of disjoint closed intervals with fast overlap detection.

    This data structure is used to track which token positions have already
    been used for insertions during training data preparation. It enables
    efficient collision detection when placing new insertions, ensuring that
    inserted sequences do not overlap with each other.

    The implementation uses a treap (tree + heap) with interval augmentation,
    providing O(log n) expected time for both overlap queries and insertions.

    Example:
        >>> intervals = IntervalSet()
        >>> intervals.add((100, 110))  # Insert tokens at positions 100-110
        >>> intervals.overlaps((105, 115))  # Check if 105-115 overlaps
        True
        >>> intervals.overlaps((200, 210))  # Check if 200-210 overlaps
        False
        >>> intervals.add((105, 115))  # This would raise ValueError (overlap)
        ValueError: Interval overlaps existing interval

    Methods:
        add(interval): Add a new interval (raises ValueError if it overlaps)
        overlaps(interval): Check if an interval overlaps any existing interval
        find_overlap(interval): Return an overlapping interval, or None
    """

    def __init__(self, it: Optional[Iterable[Interval]] = None):
        """
        Initialize an IntervalSet, optionally from an iterable of intervals.

        Args:
            it: Optional iterable of (lo, hi) tuples to add initially
        """
        self._root: Optional[_Node] = None
        if it:
            for lo, hi in it:
                self.add((lo, hi))

    def _rot_r(self, y: _Node) -> _Node:
        x = y.left
        y.left = x.right
        x.right = y
        y.recalc()
        x.recalc()
        return x

    def _rot_l(self, x: _Node) -> _Node:
        y = x.right
        x.right = y.left
        y.left = x
        x.recalc()
        y.recalc()
        return y

    def _insert(self, t: Optional[_Node], lo: int, hi: int) -> _Node:
        if not t:
            return _Node(lo, hi)
        if lo < t.lo:
            t.left = self._insert(t.left, lo, hi)
            if t.left.prio < t.prio:
                t = self._rot_r(t)
        else:
            t.right = self._insert(t.right, lo, hi)
            if t.right.prio < t.prio:
                t = self._rot_l(t)
        t.recalc()
        return t

    def add(self, iv: Interval) -> None:
        """
        Add a new interval to the set.

        Args:
            iv: A tuple (lo, hi) representing the closed interval [lo, hi]

        Raises:
            ValueError: If lo > hi or if the interval overlaps an existing one
        """
        lo, hi = iv
        if hi < lo:
            raise ValueError("Interval must satisfy lo <= hi")
        if self.overlaps(iv):
            raise ValueError("Interval overlaps existing interval")
        self._root = self._insert(self._root, lo, hi)

    def overlaps(self, iv: Interval) -> bool:
        """
        Check if an interval overlaps any existing interval in the set.

        Args:
            iv: A tuple (lo, hi) representing the closed interval to check

        Returns:
            True if the interval overlaps an existing interval, False otherwise
        """
        lo, hi = iv
        t = self._root
        while t:
            if t.left and t.left.max_end >= lo:
                t = t.left
                continue
            if _overlaps_closed((t.lo, t.hi), (lo, hi)):
                return True
            t = t.right
        return False

    def find_overlap(self, iv: Interval) -> Optional[Interval]:
        """
        Find and return an interval that overlaps the given interval.

        Args:
            iv: A tuple (lo, hi) representing the closed interval to check

        Returns:
            A tuple (lo, hi) of an overlapping interval, or None if no overlap
        """
        lo, hi = iv
        t = self._root
        while t:
            if t.left and t.left.max_end >= lo:
                t = t.left
                continue
            if _overlaps_closed((t.lo, t.hi), (lo, hi)):
                return (t.lo, t.hi)
            t = t.right
        return None

    def __len__(self) -> int:
        """Return the number of intervals in the set."""
        def cnt(n: Optional[_Node]) -> int:
            return 0 if n is None else 1 + cnt(n.left) + cnt(n.right)
        return cnt(self._root)

    def to_list(self) -> list[Interval]:
        """Return all intervals as a sorted list of (lo, hi) tuples."""
        out: list[Interval] = []

        def _inorder(n: Optional[_Node]):
            if not n:
                return
            _inorder(n.left)
            out.append((n.lo, n.hi))
            _inorder(n.right)

        _inorder(self._root)
        return out

    def __hash__(self) -> int:
        """
        Compute a stable hash of the current intervals.

        The hash is based on the sorted list of intervals, so it's independent
        of insertion order and tree structure.
        """
        intervals = self.to_list()
        intervals_str = str(intervals)
        hash_bytes = hashlib.sha256(intervals_str.encode('utf-8')).digest()
        return int.from_bytes(hash_bytes[:8], 'big', signed=True)

    def hash_fast(self) -> int:
        """
        Alternative faster hash using Python's built-in hash.

        Stable within a single Python process, but may vary across processes.
        """
        return hash(tuple(self.to_list()))

    def __eq__(self, other) -> bool:
        """Check if two IntervalSets contain the same intervals."""
        if not isinstance(other, IntervalSet):
            return False
        return self.to_list() == other.to_list()

    def __repr__(self) -> str:
        return f"IntervalSet({self.to_list()})"


# ---------------------------------------------------------------------------
# Token insertion functions
# ---------------------------------------------------------------------------


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
    logger.debug(f"Minimum sequence length: {min_length}")
    logger.debug(f"Maximum sequence length: {max_length}")

    if min_length != max_length:
        second_max_length = sorted(set(len(tokens) for tokens in token_sequences))[-2]
        logger.debug(f"Second maximum sequence length: {second_max_length}")

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

    if num_overly_long_sequences > 0:
        logger.warning(f"Dropped {num_overly_long_sequences} overly long sequences (longer than {sequence_length} tokens).")
    if num_empty_sequences > 0:
        logger.warning(f"Dropped {num_empty_sequences} empty sequences.")

    if eos_wrapped_sequences:
        min_length = min(len(tokens) for tokens in eos_wrapped_sequences)
        max_length = max(len(tokens) for tokens in eos_wrapped_sequences)
        logger.debug(f"Minimum sequence length after wrapping: {min_length}")
        logger.debug(f"Maximum sequence length after wrapping: {max_length}")

        if min_length != max_length:
            second_max_length = sorted(set(len(tokens) for tokens in eos_wrapped_sequences))[-2]
            logger.debug(f"Second maximum sequence length after wrapping: {second_max_length}")

    return eos_wrapped_sequences


def add_explicit_insertions(
    token_sequences: list[list[int]],
    positions: list[int],
    existing_insertions: Optional[IntervalSet] = None,
) -> tuple[dict[int, list[int]], IntervalSet]:
    """
    Add token sequences at explicit positions.

    Args:
        token_sequences: List of token ID lists to insert
        positions: List of global token positions (one per sequence)
        existing_insertions: IntervalSet tracking already-used positions

    Returns:
        Tuple of (insert_dict, updated_existing_insertions)
        insert_dict maps global_position -> token_sequence

    Raises:
        ValueError: If a position overlaps with an existing insertion
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
            raise ValueError(
                f"Insertion at position {pos} (length {len(tokens)}) overlaps with existing insertion"
            )

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

    # Validate that start_idx and end_idx are aligned to sequence boundaries
    if start_idx % sequence_length != 0:
        raise ValueError(
            f"start_idx ({start_idx}) must be a multiple of sequence_length ({sequence_length})"
        )
    if end_idx % sequence_length != 0:
        raise ValueError(
            f"end_idx ({end_idx}) must be a multiple of sequence_length ({sequence_length})"
        )

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
                logger.warning(
                    "Too many collisions while inserting sequences. "
                    "Consider adjusting the range or the number of sequences."
                )
                break

    total_inserted_tokens = sum(len(seq) for seq in insert_dict.values())
    logger.debug(f"Total number of inserted tokens: {total_inserted_tokens}")
    logger.debug(f"Avoided collisions while inserting sequences: {num_collisions}")

    return insert_dict, existing_insertions


def convert_insert_dict_to_index_map(
    insert_dict: dict[int, list[int]],
    num_index_tokens: int,
    split_across_boundaries: bool = True,
) -> dict[int, list[tuple[int, list[int]]]]:
    """
    Convert an insert_dict to the index-based format used by InsertionMapWriter.

    This function transforms a dictionary mapping global token positions to token
    sequences into a format that groups insertions by index (e.g., sequence index
    or batch index), with local positions within each index.

    Args:
        insert_dict: Maps global token position -> token sequence to insert.
            Example: {5: [1,2,3], 4096: [4,5,6]} means insert [1,2,3] at global
            position 5 and [4,5,6] at global position 4096.
        num_index_tokens: Number of tokens per index unit. For example:
            - sequence_length (e.g., 4096) for per-sequence indexing
            - batch_size * sequence_length for per-batch indexing
        split_across_boundaries: If True, insertions that cross index boundaries
            are split into multiple entries. If False, raises ValueError when
            an insertion would cross a boundary.

    Returns:
        Dictionary mapping index -> list of (local_position, token_ids) tuples.
        This format is compatible with InsertionMapWriter.write_dict().

        Example with num_index_tokens=4096:
            Input:  {5: [1,2,3], 4096: [4,5,6], 8190: [7,8,9,10,11,12]}
            Output: {
                0: [(5, [1,2,3])],
                1: [(0, [4,5,6])],
                2: [(4094, [7,8,9,10,11,12])]  # if split_across_boundaries=False
                # OR if split_across_boundaries=True and tokens cross boundary:
                # 1: [(4094, [7,8])], 2: [(0, [9,10,11,12])]
            }

    Raises:
        ValueError: If split_across_boundaries=False and an insertion crosses
            an index boundary.

    Example:
        >>> insert_dict = {5: [1, 2, 3], 4100: [4, 5]}
        >>> result = convert_insert_dict_to_index_map(insert_dict, num_index_tokens=4096)
        >>> result
        {0: [(5, [1, 2, 3])], 1: [(4, [4, 5])]}
    """
    if not insert_dict:
        return {}

    if num_index_tokens <= 0:
        raise ValueError(f"num_index_tokens must be positive, got {num_index_tokens}")

    index_map: dict[int, list[tuple[int, list[int]]]] = {}
    num_splits = 0

    for global_pos, tokens in insert_dict.items():
        if not tokens:
            continue

        index = global_pos // num_index_tokens
        local_pos = global_pos % num_index_tokens
        remaining_tokens = tokens

        # Check if splitting is needed
        if local_pos + len(remaining_tokens) > num_index_tokens:
            if not split_across_boundaries:
                raise ValueError(
                    f"Insertion at global position {global_pos} with {len(tokens)} tokens "
                    f"crosses index boundary (num_index_tokens={num_index_tokens}). "
                    f"Set split_across_boundaries=True to allow splitting."
                )

            # Split tokens across index boundaries
            while local_pos + len(remaining_tokens) > num_index_tokens:
                tokens_for_this_index = remaining_tokens[:num_index_tokens - local_pos]
                if index not in index_map:
                    index_map[index] = []
                index_map[index].append((local_pos, tokens_for_this_index))

                remaining_tokens = remaining_tokens[num_index_tokens - local_pos:]
                local_pos = 0
                index += 1
                num_splits += 1

        # Add remaining tokens (or all tokens if no split needed)
        if remaining_tokens:
            if index not in index_map:
                index_map[index] = []
            index_map[index].append((local_pos, remaining_tokens))

    if num_splits > 0:
        logger.debug(f"Split {num_splits} insertions across index boundaries "
                     f"(num_index_tokens={num_index_tokens}).")

    return index_map

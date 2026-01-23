"""
Tests for token_insertion module.

Focuses on convert_insert_dict_to_index_map function.
"""

import pytest

from pretrain_experiments.token_insertion import (
    convert_insert_dict_to_index_map,
    IntervalSet,
)


class TestConvertInsertDictToIndexMapBasic:
    """Test basic conversion functionality."""

    def test_empty_dict_returns_empty(self):
        """Empty insert_dict should return empty result."""
        result = convert_insert_dict_to_index_map({}, num_index_tokens=4096)
        assert result == {}

    def test_single_insertion_within_first_index(self):
        """Single insertion at position 5 with num_index_tokens=4096 -> index 0."""
        insert_dict = {5: [1, 2, 3]}
        result = convert_insert_dict_to_index_map(insert_dict, num_index_tokens=4096)

        assert result == {0: [(5, [1, 2, 3])]}

    def test_single_insertion_in_second_index(self):
        """Insertion at position 4100 with num_index_tokens=4096 -> index 1, local_pos 4."""
        insert_dict = {4100: [1, 2, 3]}
        result = convert_insert_dict_to_index_map(insert_dict, num_index_tokens=4096)

        assert result == {1: [(4, [1, 2, 3])]}

    def test_multiple_insertions_same_index(self):
        """Multiple insertions in the same index should be grouped."""
        insert_dict = {
            10: [1, 2],
            100: [3, 4, 5],
        }
        result = convert_insert_dict_to_index_map(insert_dict, num_index_tokens=4096)

        assert 0 in result
        assert len(result[0]) == 2
        # Check both insertions are present (order may vary due to dict iteration)
        insertions = {pos: tokens for pos, tokens in result[0]}
        assert insertions[10] == [1, 2]
        assert insertions[100] == [3, 4, 5]

    def test_insertions_across_multiple_indices(self):
        """Insertions in different indices should be separated."""
        insert_dict = {
            5: [1, 2, 3],       # index 0, local_pos 5
            4096: [4, 5, 6],    # index 1, local_pos 0
            8200: [7, 8],       # index 2, local_pos 8
        }
        result = convert_insert_dict_to_index_map(insert_dict, num_index_tokens=4096)

        assert result[0] == [(5, [1, 2, 3])]
        assert result[1] == [(0, [4, 5, 6])]
        assert result[2] == [(8, [7, 8])]

    def test_empty_token_sequence_skipped(self):
        """Empty token sequences should be skipped."""
        insert_dict = {5: [], 10: [1, 2]}
        result = convert_insert_dict_to_index_map(insert_dict, num_index_tokens=4096)

        assert result == {0: [(10, [1, 2])]}

    def test_exact_boundary_position(self):
        """Insertion exactly at index boundary."""
        insert_dict = {4096: [1, 2, 3]}  # Exactly at start of index 1
        result = convert_insert_dict_to_index_map(insert_dict, num_index_tokens=4096)

        assert result == {1: [(0, [1, 2, 3])]}


class TestConvertInsertDictToIndexMapSplitting:
    """Test splitting behavior when insertions cross boundaries."""

    def test_split_across_two_indices(self):
        """Insertion crossing one boundary should be split into two parts."""
        # Position 4094, 5 tokens -> crosses into index 1
        insert_dict = {4094: [1, 2, 3, 4, 5]}
        result = convert_insert_dict_to_index_map(
            insert_dict, num_index_tokens=4096, split_across_boundaries=True
        )

        # Should have entries in both index 0 and 1
        assert 0 in result
        assert 1 in result

        # Index 0: position 4094, tokens [1, 2] (2 tokens fit before boundary)
        assert result[0] == [(4094, [1, 2])]

        # Index 1: position 0, tokens [3, 4, 5] (remaining 3 tokens)
        assert result[1] == [(0, [3, 4, 5])]

    def test_split_across_three_indices(self):
        """Long insertion crossing multiple boundaries."""
        # Position 4094, 10 tokens with num_index_tokens=4
        # This means: pos 4094 in index 1023, crosses into 1024 and 1025
        insert_dict = {4094: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        result = convert_insert_dict_to_index_map(
            insert_dict, num_index_tokens=4, split_across_boundaries=True
        )

        # Position 4094 / 4 = index 1023, local_pos 2
        # 2 tokens fit in index 1023, then 4 in 1024, then 4 in 1025
        assert 1023 in result
        assert 1024 in result
        assert 1025 in result

        assert result[1023] == [(2, [1, 2])]
        assert result[1024] == [(0, [3, 4, 5, 6])]
        assert result[1025] == [(0, [7, 8, 9, 10])]

    def test_no_split_raises_error(self):
        """split_across_boundaries=False should raise error on boundary crossing."""
        insert_dict = {4094: [1, 2, 3, 4, 5]}  # Crosses boundary

        with pytest.raises(ValueError, match="crosses index boundary"):
            convert_insert_dict_to_index_map(
                insert_dict, num_index_tokens=4096, split_across_boundaries=False
            )

    def test_no_split_when_fits(self):
        """split_across_boundaries=False should work when no splitting needed."""
        insert_dict = {10: [1, 2, 3]}  # Fits entirely in index 0
        result = convert_insert_dict_to_index_map(
            insert_dict, num_index_tokens=4096, split_across_boundaries=False
        )

        assert result == {0: [(10, [1, 2, 3])]}

    def test_split_exactly_fills_index(self):
        """Insertion that exactly fills remaining space in an index."""
        # Position 4094, 2 tokens -> exactly fills index 0
        insert_dict = {4094: [1, 2]}
        result = convert_insert_dict_to_index_map(
            insert_dict, num_index_tokens=4096, split_across_boundaries=True
        )

        # Should not split, all tokens fit
        assert result == {0: [(4094, [1, 2])]}


class TestConvertInsertDictToIndexMapEdgeCases:
    """Test edge cases and special scenarios."""

    def test_zero_position(self):
        """Insertion at position 0."""
        insert_dict = {0: [1, 2, 3]}
        result = convert_insert_dict_to_index_map(insert_dict, num_index_tokens=4096)

        assert result == {0: [(0, [1, 2, 3])]}

    def test_large_position(self):
        """Insertion at a large position."""
        pos = 1_000_000
        insert_dict = {pos: [1, 2]}
        result = convert_insert_dict_to_index_map(insert_dict, num_index_tokens=4096)

        expected_index = pos // 4096
        expected_local = pos % 4096
        assert result == {expected_index: [(expected_local, [1, 2])]}

    def test_single_token_sequence(self):
        """Single token in sequence."""
        insert_dict = {100: [42]}
        result = convert_insert_dict_to_index_map(insert_dict, num_index_tokens=4096)

        assert result == {0: [(100, [42])]}

    def test_num_index_tokens_one(self):
        """Edge case: num_index_tokens=1 means each position is its own index."""
        insert_dict = {5: [1], 10: [2]}
        result = convert_insert_dict_to_index_map(insert_dict, num_index_tokens=1)

        assert result == {5: [(0, [1])], 10: [(0, [2])]}

    def test_num_index_tokens_one_with_split(self):
        """num_index_tokens=1 forces split of multi-token sequences."""
        insert_dict = {5: [1, 2, 3]}
        result = convert_insert_dict_to_index_map(
            insert_dict, num_index_tokens=1, split_across_boundaries=True
        )

        assert result == {5: [(0, [1])], 6: [(0, [2])], 7: [(0, [3])]}

    def test_invalid_num_index_tokens_zero(self):
        """num_index_tokens=0 should raise error."""
        with pytest.raises(ValueError, match="must be positive"):
            convert_insert_dict_to_index_map({5: [1, 2]}, num_index_tokens=0)

    def test_invalid_num_index_tokens_negative(self):
        """Negative num_index_tokens should raise error."""
        with pytest.raises(ValueError, match="must be positive"):
            convert_insert_dict_to_index_map({5: [1, 2]}, num_index_tokens=-1)

    def test_preserves_token_values(self):
        """Token values should be preserved exactly."""
        tokens = [100257, 33717, 71351, 396, 12203]
        insert_dict = {42: tokens}
        result = convert_insert_dict_to_index_map(insert_dict, num_index_tokens=4096)

        assert result[0][0][1] == tokens


class TestConvertInsertDictToIndexMapIntegration:
    """Integration tests with InsertionMapWriter format."""

    def test_output_compatible_with_insertion_map_writer(self, tmp_path):
        """Output should be directly usable with InsertionMapWriter."""
        from pretrain_experiments.insertion_map import (
            InsertionMapWriter,
            InsertionMapReader,
        )

        insert_dict = {
            5: [1, 2, 3],
            4100: [4, 5, 6],
            8200: [7, 8, 9, 10],
        }

        # Convert
        index_map = convert_insert_dict_to_index_map(
            insert_dict, num_index_tokens=4096
        )

        # Write using InsertionMapWriter
        file_path = tmp_path / "test.h5"
        writer = InsertionMapWriter(str(file_path))
        writer.write_dict(index_map)

        # Read back and verify
        with InsertionMapReader(str(file_path)) as reader:
            assert reader.load(0) == [(5, [1, 2, 3])]
            assert reader.load(1) == [(4, [4, 5, 6])]
            assert reader.load(2) == [(8, [7, 8, 9, 10])]

    def test_split_output_compatible_with_insertion_map(self, tmp_path):
        """Split output should also work with InsertionMapWriter."""
        from pretrain_experiments.insertion_map import (
            InsertionMapWriter,
            InsertionMapReader,
        )

        # Insertion that crosses boundary
        insert_dict = {4094: [1, 2, 3, 4, 5]}

        # Convert with splitting
        index_map = convert_insert_dict_to_index_map(
            insert_dict, num_index_tokens=4096, split_across_boundaries=True
        )

        # Write and read back
        file_path = tmp_path / "test.h5"
        writer = InsertionMapWriter(str(file_path))
        writer.write_dict(index_map)

        with InsertionMapReader(str(file_path)) as reader:
            assert reader.load(0) == [(4094, [1, 2])]
            assert reader.load(1) == [(0, [3, 4, 5])]

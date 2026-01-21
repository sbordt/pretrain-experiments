"""
Tests for InsertionMapReader and InsertionMapWriter.

Tests cover both simple and optimized HDF5 formats.
"""

import pytest
import h5py
import numpy as np
from pathlib import Path

from pretrain_experiments.insertion_map import InsertionMapReader, InsertionMapWriter


class TestInsertionMapWriterBasicOperations:
    """Test InsertionMapWriter basic write and read operations."""

    def test_write_dict_creates_file(self, tmp_path, sample_insertion_data):
        """write_dict should create a new HDF5 file."""
        file_path = tmp_path / "test.h5"
        writer = InsertionMapWriter(str(file_path))
        writer.write_dict(sample_insertion_data)

        assert file_path.exists()

    def test_write_dict_stores_correct_keys(self, tmp_path, sample_insertion_data):
        """write_dict should store all indices."""
        file_path = tmp_path / "test.h5"
        writer = InsertionMapWriter(str(file_path))
        writer.write_dict(sample_insertion_data)

        with h5py.File(file_path, "r") as f:
            keys = set(f["keys"][:].tolist())
            assert keys == {1, 2, 3}

    def test_write_dict_overwrites_existing(self, tmp_path, sample_insertion_data):
        """write_dict with mode='w' should overwrite existing data."""
        file_path = tmp_path / "test.h5"
        writer = InsertionMapWriter(str(file_path))

        # Write initial data
        writer.write_dict(sample_insertion_data)

        # Overwrite with new data
        new_data = {100: [(5, [1, 2])]}
        writer.write_dict(new_data, mode="w")

        # Should only have the new key
        assert writer.get_indices() == [100]

    def test_read_dict_returns_all_data(self, tmp_path, sample_insertion_data):
        """read_dict should return all stored data."""
        file_path = tmp_path / "test.h5"
        writer = InsertionMapWriter(str(file_path))
        writer.write_dict(sample_insertion_data)

        result = writer.read_dict()
        assert result == sample_insertion_data

    def test_read_index_returns_single_index(self, tmp_path, sample_insertion_data):
        """read_index should return data for a single index."""
        file_path = tmp_path / "test.h5"
        writer = InsertionMapWriter(str(file_path))
        writer.write_dict(sample_insertion_data)

        result = writer.read_index(1)
        assert result == [(10, [1, 2, 3]), (11, [4, 5])]

    def test_read_index_nonexistent_returns_none(self, tmp_path, sample_insertion_data):
        """read_index should return None for nonexistent index."""
        file_path = tmp_path / "test.h5"
        writer = InsertionMapWriter(str(file_path))
        writer.write_dict(sample_insertion_data)

        result = writer.read_index(999)
        assert result is None

    def test_get_indices_returns_all_keys(self, tmp_path, sample_insertion_data):
        """get_indices should return list of all indices."""
        file_path = tmp_path / "test.h5"
        writer = InsertionMapWriter(str(file_path))
        writer.write_dict(sample_insertion_data)

        indices = writer.get_indices()
        assert sorted(indices) == [1, 2, 3]

    def test_index_exists_true_for_existing(self, tmp_path, sample_insertion_data):
        """index_exists should return True for existing index."""
        file_path = tmp_path / "test.h5"
        writer = InsertionMapWriter(str(file_path))
        writer.write_dict(sample_insertion_data)

        assert writer.index_exists(1) is True
        assert writer.index_exists(2) is True

    def test_index_exists_false_for_nonexistent(self, tmp_path, sample_insertion_data):
        """index_exists should return False for nonexistent index."""
        file_path = tmp_path / "test.h5"
        writer = InsertionMapWriter(str(file_path))
        writer.write_dict(sample_insertion_data)

        assert writer.index_exists(999) is False


class TestInsertionMapWriterAppendOperations:
    """Test InsertionMapWriter append operations."""

    def test_append_dict_creates_file_if_missing(self, tmp_path):
        """append_dict should create file if it doesn't exist."""
        file_path = tmp_path / "test.h5"
        writer = InsertionMapWriter(str(file_path))

        data = {1: [(10, [1, 2, 3])]}
        writer.append_dict(data)

        assert file_path.exists()
        assert writer.read_index(1) == [(10, [1, 2, 3])]

    def test_append_dict_adds_new_index(self, tmp_path, sample_insertion_data):
        """append_dict should add new indices to existing file."""
        file_path = tmp_path / "test.h5"
        writer = InsertionMapWriter(str(file_path))
        writer.write_dict(sample_insertion_data)

        # Append new index
        new_data = {4: [(40, [100, 101])]}
        writer.append_dict(new_data)

        indices = writer.get_indices()
        assert sorted(indices) == [1, 2, 3, 4]
        assert writer.read_index(4) == [(40, [100, 101])]

    def test_append_dict_extends_existing_index(self, tmp_path, sample_insertion_data):
        """append_dict should add insertions to existing index."""
        file_path = tmp_path / "test.h5"
        writer = InsertionMapWriter(str(file_path))
        writer.write_dict(sample_insertion_data)

        # Append to existing index 1
        new_data = {1: [(12, [100, 101])]}
        writer.append_dict(new_data)

        result = writer.read_index(1)
        assert result == [(10, [1, 2, 3]), (11, [4, 5]), (12, [100, 101])]

    def test_append_dict_mixed_new_and_existing(
        self, tmp_path, sample_insertion_data, sample_insertion_data_extra
    ):
        """append_dict should handle both new and existing indices."""
        file_path = tmp_path / "test.h5"
        writer = InsertionMapWriter(str(file_path))
        writer.write_dict(sample_insertion_data)

        writer.append_dict(sample_insertion_data_extra)

        # Check existing index was extended
        result_1 = writer.read_index(1)
        assert len(result_1) == 3  # Original 2 + 1 new

        # Check new index was added
        result_4 = writer.read_index(4)
        assert result_4 == [(40, [200, 201, 202])]


class TestInsertionMapWriterOptimizedFormat:
    """Test InsertionMapWriter.save_optimized()."""

    def test_save_optimized_creates_file(self, tmp_path, sample_insertion_data):
        """save_optimized should create a new HDF5 file."""
        simple_path = tmp_path / "simple.h5"
        optimized_path = tmp_path / "optimized.h5"

        writer = InsertionMapWriter(str(simple_path))
        writer.write_dict(sample_insertion_data)
        writer.save_optimized(str(optimized_path))

        assert optimized_path.exists()

    def test_save_optimized_has_correct_structure(self, tmp_path, sample_insertion_data):
        """save_optimized should create file with offset arrays."""
        simple_path = tmp_path / "simple.h5"
        optimized_path = tmp_path / "optimized.h5"

        writer = InsertionMapWriter(str(simple_path))
        writer.write_dict(sample_insertion_data)
        writer.save_optimized(str(optimized_path))

        with h5py.File(optimized_path, "r") as f:
            assert "keys" in f
            assert "tuple_offsets" in f
            assert "positions" in f
            assert "token_offsets" in f
            assert "tokens" in f

    def test_save_optimized_preserves_data(self, tmp_path, sample_insertion_data):
        """save_optimized should preserve all data."""
        simple_path = tmp_path / "simple.h5"
        optimized_path = tmp_path / "optimized.h5"

        writer = InsertionMapWriter(str(simple_path))
        writer.write_dict(sample_insertion_data)
        writer.save_optimized(str(optimized_path))

        # Read from optimized and compare
        reader = InsertionMapReader(str(optimized_path))
        for index, expected in sample_insertion_data.items():
            assert reader.load(index) == expected
        reader.close()

    def test_save_optimized_can_be_recreated(self, tmp_path, sample_insertion_data):
        """save_optimized can overwrite existing optimized file."""
        simple_path = tmp_path / "simple.h5"
        optimized_path = tmp_path / "optimized.h5"

        writer = InsertionMapWriter(str(simple_path))
        writer.write_dict(sample_insertion_data)
        writer.save_optimized(str(optimized_path))

        # Append and re-optimize
        writer.append_dict({4: [(40, [100])]})
        writer.save_optimized(str(optimized_path))

        reader = InsertionMapReader(str(optimized_path))
        assert 4 in reader
        assert reader.load(4) == [(40, [100])]
        reader.close()


class TestInsertionMapReaderSimpleFormat:
    """Test InsertionMapReader with simple HDF5 format."""

    def test_reader_initializes(self, simple_format_file):
        """Reader should initialize with simple format file."""
        reader = InsertionMapReader(str(simple_format_file))
        assert len(reader) == 3
        reader.close()

    def test_load_returns_correct_data(self, simple_format_file, sample_insertion_data):
        """load should return correct insertion data."""
        reader = InsertionMapReader(str(simple_format_file))

        for index, expected in sample_insertion_data.items():
            assert reader.load(index) == expected

        reader.close()

    def test_load_nonexistent_returns_none(self, simple_format_file):
        """load should return None for nonexistent index."""
        reader = InsertionMapReader(str(simple_format_file))
        assert reader.load(999) is None
        reader.close()

    def test_has_index_returns_correct_value(self, simple_format_file):
        """has_index should return correct boolean."""
        reader = InsertionMapReader(str(simple_format_file))

        assert reader.has_index(1) is True
        assert reader.has_index(2) is True
        assert reader.has_index(999) is False

        reader.close()

    def test_contains_operator(self, simple_format_file):
        """'in' operator should work for index checking."""
        reader = InsertionMapReader(str(simple_format_file))

        assert 1 in reader
        assert 2 in reader
        assert 999 not in reader

        reader.close()

    def test_len_returns_correct_count(self, simple_format_file):
        """len() should return number of indices."""
        reader = InsertionMapReader(str(simple_format_file))
        assert len(reader) == 3
        reader.close()

    def test_get_all_indices(self, simple_format_file):
        """get_all_indices should return all indices."""
        reader = InsertionMapReader(str(simple_format_file))
        indices = reader.get_all_indices()
        assert sorted(indices) == [1, 2, 3]
        reader.close()


class TestInsertionMapReaderOptimizedFormat:
    """Test InsertionMapReader with optimized HDF5 format."""

    def test_reader_initializes(self, optimized_format_file):
        """Reader should initialize with optimized format file."""
        reader = InsertionMapReader(str(optimized_format_file))
        assert len(reader) == 3
        reader.close()

    def test_load_returns_correct_data(
        self, optimized_format_file, sample_insertion_data
    ):
        """load should return correct insertion data from optimized format."""
        reader = InsertionMapReader(str(optimized_format_file))

        for index, expected in sample_insertion_data.items():
            assert reader.load(index) == expected

        reader.close()

    def test_load_nonexistent_returns_none(self, optimized_format_file):
        """load should return None for nonexistent index."""
        reader = InsertionMapReader(str(optimized_format_file))
        assert reader.load(999) is None
        reader.close()

    def test_has_index_returns_correct_value(self, optimized_format_file):
        """has_index should return correct boolean."""
        reader = InsertionMapReader(str(optimized_format_file))

        assert reader.has_index(1) is True
        assert reader.has_index(999) is False

        reader.close()

    def test_contains_operator(self, optimized_format_file):
        """'in' operator should work for index checking."""
        reader = InsertionMapReader(str(optimized_format_file))

        assert 1 in reader
        assert 999 not in reader

        reader.close()


class TestInsertionMapReaderContextManager:
    """Test InsertionMapReader context manager support."""

    def test_context_manager_opens_and_closes(self, simple_format_file):
        """Context manager should properly open and close file."""
        with InsertionMapReader(str(simple_format_file)) as reader:
            assert 1 in reader
            data = reader.load(1)
            assert data is not None

    def test_context_manager_with_optimized(self, optimized_format_file):
        """Context manager should work with optimized format."""
        with InsertionMapReader(str(optimized_format_file)) as reader:
            assert 1 in reader
            data = reader.load(1)
            assert data is not None


class TestInsertionMapRoundTrip:
    """Test data integrity through write and read cycles."""

    def test_simple_format_roundtrip(self, tmp_path, sample_insertion_data):
        """Data should survive write -> read cycle in simple format."""
        file_path = tmp_path / "test.h5"

        # Write
        writer = InsertionMapWriter(str(file_path))
        writer.write_dict(sample_insertion_data)

        # Read with reader
        with InsertionMapReader(str(file_path)) as reader:
            for index, expected in sample_insertion_data.items():
                assert reader.load(index) == expected

    def test_optimized_format_roundtrip(self, tmp_path, sample_insertion_data):
        """Data should survive write -> optimize -> read cycle."""
        simple_path = tmp_path / "simple.h5"
        optimized_path = tmp_path / "optimized.h5"

        # Write and optimize
        writer = InsertionMapWriter(str(simple_path))
        writer.write_dict(sample_insertion_data)
        writer.save_optimized(str(optimized_path))

        # Read from optimized
        with InsertionMapReader(str(optimized_path)) as reader:
            for index, expected in sample_insertion_data.items():
                assert reader.load(index) == expected

    def test_simple_and_optimized_equivalent(self, tmp_path, sample_insertion_data):
        """Simple and optimized formats should return identical data."""
        simple_path = tmp_path / "simple.h5"
        optimized_path = tmp_path / "optimized.h5"

        writer = InsertionMapWriter(str(simple_path))
        writer.write_dict(sample_insertion_data)
        writer.save_optimized(str(optimized_path))

        with InsertionMapReader(str(simple_path)) as simple_reader:
            with InsertionMapReader(str(optimized_path)) as opt_reader:
                # Same indices
                assert (
                    simple_reader.get_all_indices() == opt_reader.get_all_indices()
                )

                # Same data for each index
                for index in simple_reader.get_all_indices():
                    assert simple_reader.load(index) == opt_reader.load(index)

    def test_append_and_reoptimize_roundtrip(self, tmp_path, sample_insertion_data):
        """Data should survive append -> re-optimize cycle."""
        simple_path = tmp_path / "simple.h5"
        optimized_path = tmp_path / "optimized.h5"

        writer = InsertionMapWriter(str(simple_path))
        writer.write_dict(sample_insertion_data)
        writer.save_optimized(str(optimized_path))

        # Append and re-optimize
        extra_data = {4: [(40, [100, 101, 102])]}
        writer.append_dict(extra_data)
        writer.save_optimized(str(optimized_path))

        # Verify all data present
        with InsertionMapReader(str(optimized_path)) as reader:
            assert len(reader) == 4
            for index, expected in sample_insertion_data.items():
                assert reader.load(index) == expected
            assert reader.load(4) == [(40, [100, 101, 102])]


class TestInsertionMapEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_token_list(self, tmp_path):
        """Should handle empty token list."""
        file_path = tmp_path / "test.h5"
        data = {1: [(10, [])]}

        writer = InsertionMapWriter(str(file_path))
        writer.write_dict(data)

        with InsertionMapReader(str(file_path)) as reader:
            assert reader.load(1) == [(10, [])]

    def test_single_token(self, tmp_path):
        """Should handle single token in list."""
        file_path = tmp_path / "test.h5"
        data = {1: [(10, [42])]}

        writer = InsertionMapWriter(str(file_path))
        writer.write_dict(data)

        with InsertionMapReader(str(file_path)) as reader:
            assert reader.load(1) == [(10, [42])]

    def test_large_token_list(self, tmp_path):
        """Should handle large token lists."""
        file_path = tmp_path / "test.h5"
        large_tokens = list(range(10000))
        data = {1: [(10, large_tokens)]}

        writer = InsertionMapWriter(str(file_path))
        writer.write_dict(data)

        with InsertionMapReader(str(file_path)) as reader:
            result = reader.load(1)
            assert result == [(10, large_tokens)]

    def test_many_insertions_per_index(self, tmp_path):
        """Should handle many insertions for single index."""
        file_path = tmp_path / "test.h5"
        insertions = [(i, [i * 10, i * 10 + 1]) for i in range(100)]
        data = {1: insertions}

        writer = InsertionMapWriter(str(file_path))
        writer.write_dict(data)

        with InsertionMapReader(str(file_path)) as reader:
            result = reader.load(1)
            assert result == insertions

    def test_many_indices(self, tmp_path):
        """Should handle many indices."""
        file_path = tmp_path / "test.h5"
        data = {i: [(i * 10, [i])] for i in range(1000)}

        writer = InsertionMapWriter(str(file_path))
        writer.write_dict(data)

        with InsertionMapReader(str(file_path)) as reader:
            assert len(reader) == 1000
            assert reader.load(500) == [(5000, [500])]

    def test_zero_index(self, tmp_path):
        """Should handle index 0."""
        file_path = tmp_path / "test.h5"
        data = {0: [(0, [1, 2, 3])]}

        writer = InsertionMapWriter(str(file_path))
        writer.write_dict(data)

        with InsertionMapReader(str(file_path)) as reader:
            assert 0 in reader
            assert reader.load(0) == [(0, [1, 2, 3])]

    def test_large_index_values(self, tmp_path):
        """Should handle large index values."""
        file_path = tmp_path / "test.h5"
        large_index = 2**31 - 1  # Max int32
        data = {large_index: [(10, [1, 2])]}

        writer = InsertionMapWriter(str(file_path))
        writer.write_dict(data)

        with InsertionMapReader(str(file_path)) as reader:
            assert large_index in reader
            assert reader.load(large_index) == [(10, [1, 2])]

    def test_optimized_format_with_edge_cases(self, tmp_path):
        """Optimized format should handle edge cases correctly."""
        simple_path = tmp_path / "simple.h5"
        optimized_path = tmp_path / "optimized.h5"

        data = {
            0: [(0, [])],  # Zero index, empty tokens
            1: [(10, [1])],  # Single token
            2: [(20, list(range(100)))],  # Many tokens
        }

        writer = InsertionMapWriter(str(simple_path))
        writer.write_dict(data)
        writer.save_optimized(str(optimized_path))

        with InsertionMapReader(str(optimized_path)) as reader:
            assert reader.load(0) == [(0, [])]
            assert reader.load(1) == [(10, [1])]
            assert reader.load(2) == [(20, list(range(100)))]

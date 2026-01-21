"""
Shared pytest fixtures for pretrain-experiments tests.
"""

import pytest
from pathlib import Path

from pretrain_experiments.insertion_map import InsertionMapWriter


@pytest.fixture
def sample_insertion_data():
    """Sample insertion data for testing.

    Structure: index -> [(position, [token_ids]), ...]
    """
    return {
        1: [(10, [1, 2, 3]), (11, [4, 5])],
        2: [(20, [6, 7, 8, 9])],
        3: [(30, [10, 11, 12])],
    }


@pytest.fixture
def sample_insertion_data_extra():
    """Additional insertion data for append tests."""
    return {
        1: [(12, [100, 101])],  # Append to existing index
        4: [(40, [200, 201, 202])],  # New index
    }


@pytest.fixture
def simple_format_file(tmp_path, sample_insertion_data):
    """Create a simple-format HDF5 file with sample data."""
    file_path = tmp_path / "simple.h5"
    writer = InsertionMapWriter(str(file_path))
    writer.write_dict(sample_insertion_data)
    return file_path


@pytest.fixture
def optimized_format_file(tmp_path, sample_insertion_data):
    """Create an optimized-format HDF5 file with sample data."""
    simple_path = tmp_path / "simple_temp.h5"
    optimized_path = tmp_path / "optimized.h5"

    writer = InsertionMapWriter(str(simple_path))
    writer.write_dict(sample_insertion_data)
    writer.save_optimized(str(optimized_path))

    return optimized_path

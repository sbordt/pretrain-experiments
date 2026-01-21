import h5py
import numpy as np
from typing import Dict, List, Tuple, Optional


class InsertionMapReader:
    """
    Reader for insertion maps stored in HDF5 format.

    Maps integer indices to insertion specifications:
        index -> [(position, [token_ids]), ...]

    Each tuple specifies where to insert tokens and which tokens to insert.

    The index can represent different concepts depending on your use case:
        - Sequence index: insertions into specific sequences in training data
        - Batch index: insertions into specific batches
        - Any other integer identifier relevant to your framework

    Supports both formats:
        - Simple format: created by InsertionMapWriter.write_dict() / append_dict()
        - Optimized format: created by InsertionMapWriter.save_optimized()

    The optimized format is recommended for large files and frequent random access.

    Optimized for random access patterns with lazy file opening
    and configurable HDF5 chunk caching.
    """

    def __init__(self, hdf5_path: str, cache_size_mb: int = 64):
        """
        Initialize the insertion map reader.

        Args:
            hdf5_path: Path to the HDF5 file (simple or optimized format)
            cache_size_mb: Cache size in MB for HDF5 chunk cache
        """
        self.hdf5_path = hdf5_path
        self.cache_size_mb = cache_size_mb

        # File handle and datasets (initialized lazily)
        self._f = None
        self._keys_array = None
        self._key_to_idx = None
        self._is_optimized = None

        # Dataset handles for optimized format
        self._tuple_offsets = None
        self._positions = None
        self._token_offsets = None
        self._tokens = None

        # Dataset handle for simple format
        self._data_group = None

        # Load indices immediately
        self._load_indices()

        print(f"InsertionMapReader initialized with {len(self._keys_array)} indices")

    def _ensure_file_open(self):
        """Lazy file opening with optimal cache settings"""
        if self._f is None:
            cache_bytes = self.cache_size_mb * 1024 * 1024
            self._f = h5py.File(
                self.hdf5_path, 'r',
                rdcc_nbytes=cache_bytes,
                rdcc_nslots=10007  # Prime number for hash efficiency
            )

            # Detect format
            if 'tuple_offsets' in self._f:
                self._is_optimized = True
                self._tuple_offsets = self._f['tuple_offsets']
                self._positions = self._f['positions']
                self._token_offsets = self._f['token_offsets']
                self._tokens = self._f['tokens']
            else:
                self._is_optimized = False
                self._data_group = self._f['data']

    def _load_indices(self):
        """Load all available indices from the HDF5 file"""
        with h5py.File(self.hdf5_path, 'r') as f:
            self._keys_array = f['keys'][:]

        # Create fast lookup dictionary
        self._key_to_idx = {int(key): idx for idx, key in enumerate(self._keys_array)}

        # Convert keys to a set for O(1) membership testing
        self._keys_set = set(int(key) for key in self._keys_array)

    def has_index(self, index: int) -> bool:
        """
        Check if an index has insertions without accessing the file.

        Args:
            index: The index to check (e.g., sequence index, batch index)

        Returns:
            True if insertions exist for this index, False otherwise
        """
        return index in self._keys_set

    def load(self, index: int) -> Optional[List[Tuple[int, List[int]]]]:
        """
        Load insertion data for a specific index.

        Args:
            index: The index to load insertions for (e.g., sequence index, batch index)

        Returns:
            List of tuples (position, [token_ids]) specifying where and what
            to insert, or None if no insertions exist for this index
        """
        if not self.has_index(index):
            return None

        self._ensure_file_open()

        if self._is_optimized:
            return self._load_optimized(index)
        else:
            return self._load_simple(index)

    def _load_optimized(self, index: int) -> List[Tuple[int, List[int]]]:
        """Load from optimized HDF5 format using offset arrays"""
        idx = self._key_to_idx[index]

        # Get tuple range for this index
        tuple_start = int(self._tuple_offsets[idx])
        tuple_end = int(self._tuple_offsets[idx + 1])

        result = []
        for t in range(tuple_start, tuple_end):
            position = int(self._positions[t])
            token_start = int(self._token_offsets[t])
            token_end = int(self._token_offsets[t + 1])
            token_ids = self._tokens[token_start:token_end].tolist()
            result.append((position, token_ids))

        return result

    def _load_simple(self, index: int) -> List[Tuple[int, List[int]]]:
        """Load from simple HDF5 format"""
        key_str = str(index)
        if key_str not in self._data_group:
            return None

        key_group = self._data_group[key_str]
        num_tuples = key_group.attrs['num_tuples']

        result = []
        for i in range(num_tuples):
            tuple_group = key_group[f'tuple_{i}']
            position = int(tuple_group.attrs['tuple_int'])
            token_ids = tuple_group['int_list'][:].tolist()
            result.append((position, token_ids))

        return result

    def get_all_indices(self) -> List[int]:
        """
        Get all indices that have insertions.

        Returns:
            List of all indices in the file
        """
        return [int(key) for key in self._keys_array]

    def __len__(self) -> int:
        """Return number of indices with insertions"""
        return len(self._keys_array)

    def __contains__(self, index: int) -> bool:
        """Support 'in' operator for checking if an index has insertions"""
        return self.has_index(index)

    def close(self):
        """Close the file handle"""
        if self._f is not None:
            self._f.close()
            self._f = None

    def __del__(self):
        """Clean up file handle"""
        self.close()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
        return False


class InsertionMapWriter:
    """
    Writer for insertion maps in HDF5 format.

    Stores mappings from integer indices to insertion specifications,
    supporting incremental writes without loading entire file into memory.

    Data format:
        Dict[index, List[Tuple[position, List[token_id]]]]

    The index can represent different concepts depending on your use case:
        - Sequence index: insertions into specific sequences in training data
        - Batch index: insertions into specific batches
        - Any other integer identifier relevant to your framework

    Each entry maps an index to a list of insertions, where each insertion
    is a tuple of (position, token_ids_to_insert).

    Typical workflow:
        writer = InsertionMapWriter("working.h5")
        writer.write_dict(dict1)
        writer.append_dict(dict2)
        writer.save_optimized("optimized.h5")  # Create optimized file for reading
        # Later...
        writer.append_dict(dict3)
        writer.save_optimized("optimized.h5")  # Re-create optimized file
    """

    def __init__(self, hdf5_path: str):
        """
        Initialize the insertion map writer.

        Args:
            hdf5_path: Path where the HDF5 file will be written (simple format)
        """
        self.hdf5_path = hdf5_path

    def write_dict(self, insertion_map: Dict[int, List[Tuple[int, List[int]]]],
                   mode: str = 'w') -> None:
        """
        Write insertion map to HDF5 file in simple format.

        Args:
            insertion_map: Dictionary mapping indices to lists of
                          (position, [token_ids]) tuples
            mode: 'w' for overwrite, 'a' for append (default 'w')
        """
        print(f"Writing {len(insertion_map)} entries to {self.hdf5_path} (mode: {mode})")
        self._optimize_for_lustre()

        with h5py.File(self.hdf5_path, mode) as f:
            if mode == 'w':
                self._write_fresh_file(f, insertion_map)
            else:
                self._append_to_file(f, insertion_map)

        print(f"Successfully wrote data to {self.hdf5_path}")

    def append_dict(self, insertion_map: Dict[int, List[Tuple[int, List[int]]]]) -> None:
        """
        Append additional insertions to existing HDF5 file without loading entire file.

        Args:
            insertion_map: New insertion data to append
        """
        print(f"Appending {len(insertion_map)} entries to {self.hdf5_path}")

        import os
        if not os.path.exists(self.hdf5_path):
            self.write_dict(insertion_map, mode='w')
            return

        with h5py.File(self.hdf5_path, 'a') as f:
            self._append_to_file(f, insertion_map)

        print(f"Successfully appended data to {self.hdf5_path}")

    def save_optimized(self, output_path: str) -> None:
        """
        Read the simple format file and write it to a new file in optimized format.

        The optimized format uses offset arrays for efficient random access:
            /keys           - int64[num_indices]: all indices
            /tuple_offsets  - int64[num_indices + 1]: where each index's tuples start
            /positions      - int32[total_tuples]: all positions flattened
            /token_offsets  - int64[total_tuples + 1]: where each tuple's tokens start
            /tokens         - int32[total_tokens]: all token_ids flattened

        Args:
            output_path: Path for the optimized HDF5 file
        """
        print(f"Creating optimized file {output_path} from {self.hdf5_path}")

        # Read all data from simple format
        with h5py.File(self.hdf5_path, 'r') as f:
            if 'keys' not in f:
                raise ValueError(f"No data in {self.hdf5_path}")

            keys = f['keys'][:]
            data_group = f['data']

            # Collect all data
            all_positions = []
            all_tokens = []
            tuple_offsets = [0]
            token_offsets = [0]

            for key in keys:
                key_str = str(int(key))
                if key_str not in data_group:
                    continue

                key_group = data_group[key_str]
                num_tuples = key_group.attrs['num_tuples']

                for i in range(num_tuples):
                    tuple_group = key_group[f'tuple_{i}']
                    position = int(tuple_group.attrs['tuple_int'])
                    token_ids = tuple_group['int_list'][:].tolist()

                    all_positions.append(position)
                    all_tokens.extend(token_ids)
                    token_offsets.append(len(all_tokens))

                tuple_offsets.append(len(all_positions))

        # Write optimized format
        self._optimize_for_lustre_path(output_path)

        with h5py.File(output_path, 'w') as f:
            f.create_dataset('keys', data=keys, dtype=np.int64)
            f.create_dataset('tuple_offsets', data=np.array(tuple_offsets, dtype=np.int64))
            f.create_dataset('positions', data=np.array(all_positions, dtype=np.int32))
            f.create_dataset('token_offsets', data=np.array(token_offsets, dtype=np.int64))
            f.create_dataset('tokens', data=np.array(all_tokens, dtype=np.int32))

        print(f"Successfully created optimized file with {len(keys)} indices, "
              f"{len(all_positions)} tuples, {len(all_tokens)} tokens")

    def _write_fresh_file(self, f, insertion_map):
        """Write data to a fresh file in simple format"""
        indices = list(insertion_map.keys())
        f.create_dataset('keys', data=np.array(indices, dtype=np.int64))

        data_group = f.create_group('data')

        for index, insertions in insertion_map.items():
            self._write_index_data(data_group, index, insertions)

    def _append_to_file(self, f, new_insertion_map):
        """Append data to existing file structure"""
        existing_keys = set()
        if 'keys' in f:
            existing_keys = set(f['keys'][:].tolist())

        if 'data' not in f:
            data_group = f.create_group('data')
        else:
            data_group = f['data']

        all_keys = existing_keys.copy()

        for index, insertions in new_insertion_map.items():
            if index in existing_keys:
                self._append_to_index_data(data_group, index, insertions)
            else:
                self._write_index_data(data_group, index, insertions)
                all_keys.add(index)

        if 'keys' in f:
            del f['keys']
        f.create_dataset('keys', data=np.array(sorted(all_keys), dtype=np.int64))

    def _write_index_data(self, data_group, index, insertions):
        """Write insertion data for a single index"""
        key_group = data_group.create_group(str(index))
        key_group.attrs['num_tuples'] = len(insertions)

        for i, (position, token_ids) in enumerate(insertions):
            tuple_group = key_group.create_group(f'tuple_{i}')
            tuple_group.attrs['tuple_int'] = position
            tuple_group.create_dataset('int_list',
                                       data=np.array(token_ids, dtype=np.int32),
                                       compression='lzf')

    def _append_to_index_data(self, data_group, index, new_insertions):
        """Append new insertions to existing index data"""
        key_str = str(index)
        key_group = data_group[key_str]

        current_num = key_group.attrs['num_tuples']

        for i, (position, token_ids) in enumerate(new_insertions):
            new_index = current_num + i
            tuple_group = key_group.create_group(f'tuple_{new_index}')
            tuple_group.attrs['tuple_int'] = position
            tuple_group.create_dataset('int_list',
                                       data=np.array(token_ids, dtype=np.int32),
                                       compression='lzf')

        key_group.attrs['num_tuples'] = current_num + len(new_insertions)

    def read_dict(self) -> Dict[int, List[Tuple[int, List[int]]]]:
        """
        Read entire insertion map from HDF5 file (simple format).

        Returns:
            Dictionary mapping indices to lists of (position, [token_ids]) tuples
        """
        print(f"Reading entire insertion map from {self.hdf5_path}")
        result = {}

        with h5py.File(self.hdf5_path, 'r') as f:
            if 'keys' not in f:
                return {}

            keys = f['keys'][:]
            data_group = f['data']

            for key in keys:
                key_str = str(key)
                if key_str in data_group:
                    key_group = data_group[key_str]
                    num_tuples = key_group.attrs['num_tuples']
                    insertions = []

                    for i in range(num_tuples):
                        tuple_group = key_group[f'tuple_{i}']
                        position = int(tuple_group.attrs['tuple_int'])
                        token_ids = tuple_group['int_list'][:].tolist()
                        insertions.append((position, token_ids))

                    result[int(key)] = insertions

        print(f"Successfully read {len(result)} entries")
        return result

    def read_index(self, index: int) -> Optional[List[Tuple[int, List[int]]]]:
        """
        Read insertions for a single index without loading entire file (simple format).

        Args:
            index: The index to read insertions for

        Returns:
            List of (position, [token_ids]) tuples, or None if not found
        """
        with h5py.File(self.hdf5_path, 'r') as f:
            if 'data' not in f:
                return None

            data_group = f['data']
            key_str = str(index)

            if key_str not in data_group:
                return None

            key_group = data_group[key_str]
            num_tuples = key_group.attrs['num_tuples']
            insertions = []

            for i in range(num_tuples):
                tuple_group = key_group[f'tuple_{i}']
                position = int(tuple_group.attrs['tuple_int'])
                token_ids = tuple_group['int_list'][:].tolist()
                insertions.append((position, token_ids))

            return insertions

    def get_indices(self) -> List[int]:
        """
        Get all indices without loading the insertion data.

        Returns:
            List of indices in the file
        """
        with h5py.File(self.hdf5_path, 'r') as f:
            if 'keys' not in f:
                return []
            return f['keys'][:].tolist()

    def index_exists(self, index: int) -> bool:
        """
        Check if an index exists in the file without loading data.

        Args:
            index: The index to check

        Returns:
            True if index exists, False otherwise
        """
        with h5py.File(self.hdf5_path, 'r') as f:
            if 'data' not in f:
                return False
            return str(index) in f['data']

    def _optimize_for_lustre(self):
        """Optimize directory for Lustre filesystem if possible"""
        self._optimize_for_lustre_path(self.hdf5_path)

    def _optimize_for_lustre_path(self, path: str):
        """Optimize directory for Lustre filesystem if possible"""
        import os
        try:
            directory = os.path.dirname(path)
            if directory and os.path.exists(directory):
                os.system(f"lfs setstripe -c 1 {directory} 2>/dev/null")
        except:
            pass


# Example usage:
if __name__ == "__main__":
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        working_file = os.path.join(tmpdir, "working.h5")
        optimized_file = os.path.join(tmpdir, "optimized.h5")

        # Create writer for incremental writes
        writer = InsertionMapWriter(working_file)

        # Initial data: sequence indices with insertions
        # For sequence 1: insert tokens [1,2,3] at position 10, and [4,5] at position 11
        # For sequence 2: insert tokens [6,7,8,9] at position 20
        initial_data = {
            1: [(10, [1, 2, 3]), (11, [4, 5])],
            2: [(20, [6, 7, 8, 9])]
        }
        writer.write_dict(initial_data)

        # Append more data
        additional_data = {
            1: [(12, [10, 11])],  # Add another insertion to index 1
            3: [(30, [12, 13, 14])]  # New index
        }
        writer.append_dict(additional_data)

        # Reader can read simple format directly
        print("Reading simple format:")
        with InsertionMapReader(working_file) as reader:
            print(f"  Index 1 insertions: {reader.load(1)}")
            print(f"  All indices: {reader.get_all_indices()}")

        # Create optimized file for efficient reading
        writer.save_optimized(optimized_file)

        # Reader can also read optimized format
        print("\nReading optimized format:")
        with InsertionMapReader(optimized_file) as reader:
            print(f"  Index 1 insertions: {reader.load(1)}")
            print(f"  All indices: {reader.get_all_indices()}")

        # Later, append more data to working file
        more_data = {
            4: [(40, [100, 101, 102])]
        }
        writer.append_dict(more_data)

        # Re-create optimized file
        writer.save_optimized(optimized_file)

        # Verify new data is accessible
        with InsertionMapReader(optimized_file) as reader:
            print(f"\nAfter adding more data:")
            print(f"  All indices: {reader.get_all_indices()}")
            print(f"  Index 4 insertions: {reader.load(4)}")

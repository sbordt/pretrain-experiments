import h5py
import numpy as np
from typing import Dict, List, Tuple, Optional

class DynamicHDF5Storage:
    """
    Dynamic HDF5 storage for Python dict structures with tuple-list values.
    Supports incremental writing without loading entire file into memory.
    """
    
    def __init__(self, hdf5_path: str):
        self.hdf5_path = hdf5_path
        
    def write_dict(self, data_dict: Dict[int, List[Tuple[int, List[int]]]], 
                   mode: str = 'w') -> None:
        """
        Write entire Python dict structure to HDF5 file.
        
        Args:
            data_dict: Dictionary where keys are integers and values are
                      lists of tuples (int, list[int])
            mode: 'w' for overwrite, 'a' for append (default 'w' for compatibility)
        """
        print(f"Writing {len(data_dict)} entries to {self.hdf5_path} (mode: {mode})")
        self._optimize_for_lustre()
        
        with h5py.File(self.hdf5_path, mode) as f:
            if mode == 'w':
                # Original write mode - overwrite everything
                self._write_fresh_file(f, data_dict)
            else:
                # Append mode - merge with existing data
                self._append_to_file(f, data_dict)
                
        print(f"Successfully wrote data to {self.hdf5_path}")
    
    def append_dict(self, data_dict: Dict[int, List[Tuple[int, List[int]]]]) -> None:
        """
        Append additional data to existing HDF5 file without loading entire file.
        
        Args:
            data_dict: New dictionary data to append
        """
        print(f"Appending {len(data_dict)} entries to {self.hdf5_path}")
        
        # Check if file exists
        import os
        if not os.path.exists(self.hdf5_path):
            # File doesn't exist, create it
            self.write_dict(data_dict, mode='w')
            return
            
        with h5py.File(self.hdf5_path, 'a') as f:
            self._append_to_file(f, data_dict)
            
        print(f"Successfully appended data to {self.hdf5_path}")
    
    def _write_fresh_file(self, f, data_dict):
        """Write data to a fresh file (original logic)"""
        # Store keys as a dataset
        keys = list(data_dict.keys())
        f.create_dataset('keys', data=np.array(keys, dtype=np.int64))
        
        # Create a group for the actual data
        data_group = f.create_group('data')
        
        # Store each key's data as a separate group
        for key, value in data_dict.items():
            self._write_key_data(data_group, key, value)
    
    def _append_to_file(self, f, new_data_dict):
        """Append data to existing file structure"""
        # Get existing keys or create empty array if doesn't exist
        existing_keys = set()
        if 'keys' in f:
            existing_keys = set(f['keys'][:].tolist())
            
        # Get or create data group
        if 'data' not in f:
            data_group = f.create_group('data')
        else:
            data_group = f['data']
        
        # Process new data
        all_keys = existing_keys.copy()
        
        for key, value in new_data_dict.items():
            if key in existing_keys:
                # Key exists - append to existing data
                self._append_to_key_data(data_group, key, value)
            else:
                # New key - create new group
                self._write_key_data(data_group, key, value)
                all_keys.add(key)
        
        # Update keys dataset
        if 'keys' in f:
            del f['keys']
        f.create_dataset('keys', data=np.array(sorted(all_keys), dtype=np.int64))
    
    def _write_key_data(self, data_group, key, value):
        """Write data for a single key"""
        key_group = data_group.create_group(str(key))
        key_group.attrs['num_tuples'] = len(value)
        
        for i, (tuple_int, int_list) in enumerate(value):
            tuple_group = key_group.create_group(f'tuple_{i}')
            tuple_group.attrs['tuple_int'] = tuple_int
            tuple_group.create_dataset('int_list',
                                     data=np.array(int_list, dtype=np.int32),
                                     compression='lzf')
    
    def _append_to_key_data(self, data_group, key, new_value):
        """Append new tuples to existing key data"""
        key_str = str(key)
        key_group = data_group[key_str]
        
        # Get current number of tuples
        current_num = key_group.attrs['num_tuples']
        
        # Add new tuples
        for i, (tuple_int, int_list) in enumerate(new_value):
            new_index = current_num + i
            tuple_group = key_group.create_group(f'tuple_{new_index}')
            tuple_group.attrs['tuple_int'] = tuple_int
            tuple_group.create_dataset('int_list',
                                     data=np.array(int_list, dtype=np.int32),
                                     compression='lzf')
        
        # Update tuple count
        key_group.attrs['num_tuples'] = current_num + len(new_value)
    
    def read_dict(self) -> Dict[int, List[Tuple[int, List[int]]]]:
        """
        Read entire Python dict structure from HDF5 file.
        """
        print(f"Reading entire dict from {self.hdf5_path}")
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
                    key_data = []
                    
                    for i in range(num_tuples):
                        tuple_group = key_group[f'tuple_{i}']
                        tuple_int = int(tuple_group.attrs['tuple_int'])
                        int_list = tuple_group['int_list'][:].tolist()
                        key_data.append((tuple_int, int_list))
                    
                    result[int(key)] = key_data
        
        print(f"Successfully read {len(result)} entries")
        return result
    
    def read_key(self, key: int) -> Optional[List[Tuple[int, List[int]]]]:
        """
        Read data for a single key without loading entire file.
        
        Args:
            key: The key to read data for
            
        Returns:
            List of tuples for the key, or None if key doesn't exist
        """
        with h5py.File(self.hdf5_path, 'r') as f:
            if 'data' not in f:
                return None
                
            data_group = f['data']
            key_str = str(key)
            
            if key_str not in data_group:
                return None
                
            key_group = data_group[key_str]
            num_tuples = key_group.attrs['num_tuples']
            key_data = []
            
            for i in range(num_tuples):
                tuple_group = key_group[f'tuple_{i}']
                tuple_int = int(tuple_group.attrs['tuple_int'])
                int_list = tuple_group['int_list'][:].tolist()
                key_data.append((tuple_int, int_list))
            
            return key_data
    
    def get_keys(self) -> List[int]:
        """
        Get all keys without loading the data.
        
        Returns:
            List of keys in the file
        """
        with h5py.File(self.hdf5_path, 'r') as f:
            if 'keys' not in f:
                return []
            return f['keys'][:].tolist()
    
    def key_exists(self, key: int) -> bool:
        """
        Check if a key exists in the file without loading data.
        
        Args:
            key: The key to check
            
        Returns:
            True if key exists, False otherwise
        """
        with h5py.File(self.hdf5_path, 'r') as f:
            if 'data' not in f:
                return False
            return str(key) in f['data']
    
    def _optimize_for_lustre(self):
        """Optimize directory for Lustre filesystem if possible"""
        import os
        try:
            directory = os.path.dirname(self.hdf5_path)
            if directory and os.path.exists(directory):
                # Try to set Lustre striping (this may fail silently)
                os.system(f"lfs setstripe -c 1 {directory} 2>/dev/null")
        except:
            pass


# Example usage:
if __name__ == "__main__":
    # Create storage instance
    storage = DynamicHDF5Storage("test_dynamic.h5")
    
    # Initial data
    initial_data = {
        1: [(10, [1, 2, 3]), (11, [4, 5])],
        2: [(20, [6, 7, 8, 9])]
    }
    
    # Write initial data
    storage.write_dict(initial_data)
    
    # Additional data to append
    additional_data = {
        1: [(12, [10, 11])],  # Append to existing key
        3: [(30, [12, 13, 14])]  # New key
    }
    
    # Append without loading entire file
    storage.append_dict(additional_data)
    
    # Read specific key
    key_1_data = storage.read_key(1)
    print(f"Key 1 data: {key_1_data}")
    
    # Get all keys
    all_keys = storage.get_keys()
    print(f"All keys: {all_keys}")
    
    # Read entire dict (for verification)
    full_data = storage.read_dict()
    print(f"Full data: {full_data}")
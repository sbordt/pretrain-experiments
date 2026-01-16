"""
flexible_config.py - Flexible argument parsing with YAML configuration and dot notation overrides.

This module provides a simple interface to load YAML configuration files
and override specific parameters using dot notation command line arguments,
with environment variable substitution support and list indexing.

Usage:
    python script.py --config config.yaml --model.path "new/path" --training.lr 0.001
    python script.py --config config.yaml --experiments.0.repetitions 0.2

Written by Claude.
"""

import argparse
import yaml
import os
import re
from typing import Dict, Any, Optional, List, Union
from pathlib import Path


def load_env_file(env_file: str) -> None:
    """Load environment variables from a file."""
    if not os.path.exists(env_file):
        raise FileNotFoundError(f"Environment file not found: {env_file}")
    
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                key = key.replace('export ', '').strip()
                value = value.strip().strip('"\'')
                os.environ[key] = value


def substitute_env_vars(content: str) -> str:
    """Substitute ${VAR} and $VAR patterns with environment variables."""
    def replace_var(match):
        var_name = match.group(1)
        env_value = os.environ.get(var_name)
        if env_value is None:
            print(f"Warning: Environment variable '{var_name}' not found, keeping placeholder")
            return match.group(0)
        return env_value
    
    # Handle ${VAR} syntax
    content = re.sub(r'\$\{([^}]+)\}', replace_var, content)
    # Handle $VAR syntax
    content = re.sub(r'\$(\w+)(?=\W|$)', replace_var, content)
    
    return content


def auto_load_env_files(config_path: str, explicit_env_file: Optional[str] = None) -> None:
    """Auto-load environment files based on config file location and naming conventions."""
    if explicit_env_file:
        load_env_file(explicit_env_file)
        return
    
    config_path = Path(config_path)
    config_dir = config_path.parent
    config_stem = config_path.stem
    
    # Look for environment files in order of preference
    auto_env_files = [
        config_dir / f"{config_stem}.env",  # config.env
        config_dir / ".env",                # .env in same directory
    ]
    
    # Add system-specific env files if SYSTEM_NAME env var is set
    system_name = os.environ.get('SYSTEM_NAME')
    if system_name:
        auto_env_files.insert(0, config_dir / f"{system_name}.env")
    
    # Load the first available auto-detected env file
    for auto_env_file in auto_env_files:
        if auto_env_file.exists():
            print(f"Auto-loading environment file: {auto_env_file}")
            load_env_file(str(auto_env_file))
            break


def deep_merge_configs(base_config: Dict[Any, Any], override_config: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Deep merge two configuration dictionaries.
    - Override config takes precedence over base config
    - Lists are concatenated (base + override)
    - Nested dictionaries are recursively merged
    """
    result = base_config.copy()
    for key, value in override_config.items():
        if key in result:
            # If both values are dictionaries, merge them recursively
            if isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge_configs(result[key], value)
            # If both values are lists, concatenate them
            elif isinstance(result[key], list) and isinstance(value, list):
                result[key] = result[key] + value
            # Otherwise, override takes precedence
            else:
                result[key] = value
        else:
            # New key from override
            result[key] = value
    return result


def load_yaml_with_includes(file_path: Union[str, Path], visited_files: set = None) -> Dict[Any, Any]:
    """
    Load a YAML file and handle 'include' directives.
    
    Args:
        file_path: Path to the YAML file to load
        visited_files: Set of already visited files to prevent circular includes
        
    Returns:
        Merged configuration dictionary
        
    Raises:
        FileNotFoundError: If included file doesn't exist
        ValueError: If circular include is detected
    """
    if visited_files is None:
        visited_files = set()
        
    file_path = Path(file_path).resolve()
    
    # Check for circular includes
    if str(file_path) in visited_files:
        raise ValueError(f"Circular include detected: {file_path}")
    visited_files.add(str(file_path))
    
    try:
        # Load and process the current YAML file with environment variable substitution
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Substitute environment variables
        content = substitute_env_vars(content)
        
        # Parse the YAML content
        config = yaml.safe_load(content) or {}
        
        # Check if there's an include directive
        if 'include' in config:
            include_file = config.pop('include')  # Remove include from config
            
            # Resolve include path relative to current file's directory
            if not os.path.isabs(include_file):
                include_file = file_path.parent / include_file
            
            # Recursively load the included file
            base_config = load_yaml_with_includes(include_file, visited_files.copy())
            
            # Merge base config with current config (current takes precedence)
            config = deep_merge_configs(base_config, config)
        
        return config
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file: {file_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file {file_path}: {e}")


def load_yaml_config(config_path: str, env_file: Optional[str] = None) -> Dict[str, Any]:
    """Load YAML configuration with environment variable substitution and includes support."""
    # Auto-load environment files
    auto_load_env_files(config_path, env_file)
    
    # Load YAML with includes support
    config = load_yaml_with_includes(config_path)
    
    return config or {}


def set_nested_value_direct(dictionary: Dict[str, Any], key_path: str, value: Any) -> None:
    """
    Set a value in an existing nested dictionary using dot notation, with list index support.
    This modifies the dictionary in place rather than creating new structures.
    """
    keys = key_path.split('.')
    current = dictionary
    
    # Navigate to the parent of the final key
    for i, key in enumerate(keys[:-1]):
        # Check if current is a list and key is a numeric index
        if isinstance(current, list):
            if not key.isdigit():
                raise ValueError(f"Cannot use non-numeric key '{key}' for list access")
            idx = int(key)
            if idx >= len(current):
                raise IndexError(f"List index {idx} out of range (list has {len(current)} elements)")
            # Move to the indexed element
            current = current[idx]
        else:
            # Regular dictionary navigation
            if key not in current:
                # Create new structure only if it doesn't exist
                # Look ahead to see if next key is numeric (indicating we need a list)
                if i + 1 < len(keys) - 1 and keys[i + 1].isdigit():
                    current[key] = []
                else:
                    current[key] = {}
            current = current[key]
    
    # Set the final value
    final_key = keys[-1]
    if isinstance(current, list):
        if not final_key.isdigit():
            raise ValueError(f"Cannot use non-numeric key '{final_key}' for list access")
        idx = int(final_key)
        if idx >= len(current):
            raise IndexError(f"List index {idx} out of range (list has {len(current)} elements)")
        current[idx] = value
    else:
        current[final_key] = value


def convert_value(value: str) -> Any:
    """Convert string value to appropriate Python type."""
    # Handle boolean strings
    if value.lower() in ['true', 'false']:
        return value.lower() == 'true'
    
    # Try to convert to number
    try:
        if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
            return int(value)
        elif '.' in value:
            return float(value)
    except ValueError:
        pass
    
    return value


def get_nested_value(dictionary: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """Get a value from a nested dictionary using dot notation, with list index support."""
    keys = key_path.split('.')
    current = dictionary
    
    try:
        for key in keys:
            if isinstance(current, list):
                if not key.isdigit():
                    return default
                current = current[int(key)]
            else:
                current = current[key]
        return current
    except (KeyError, TypeError, IndexError, ValueError):
        return default


class FlexibleConfig(dict):
    """Enhanced dictionary with dot notation access for nested values and list indexing."""
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a value using dot notation.
        
        Args:
            key_path: Dot-separated key path (e.g., 'model.name' or 'experiments.0.name')
            default: Default value if key not found
            
        Returns:
            The value at the specified path or default if not found
            
        Examples:
            config.get('model.name')
            config.get('training.lr', 0.001)
            config.get('experiments.0.repetitions', 1.0)
        """
        return get_nested_value(self, key_path, default)
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set a value using dot notation.
        
        Args:
            key_path: Dot-separated key path (e.g., 'model.name' or 'experiments.0.repetitions')
            value: Value to set
        """
        set_nested_value_direct(self, key_path, value)
    
    def has(self, key_path: str) -> bool:
        """
        Check if a key path exists in the configuration.
        
        Args:
            key_path: Dot-separated key path (e.g., 'model.name' or 'experiments.0.name')
            
        Returns:
            True if the key path exists, False otherwise
        """
        keys = key_path.split('.')
        current = self
        
        try:
            for key in keys:
                if isinstance(current, list):
                    if not key.isdigit():
                        return False
                    current = current[int(key)]
                else:
                    current = current[key]
            return True
        except (KeyError, TypeError, IndexError, ValueError):
            return False


def parse_config_with_overrides(description: Optional[str] = None) -> FlexibleConfig:
    """
    Parse configuration from YAML file with command line dot notation overrides.
    
    Args:
        description: Description for the argument parser
        
    Returns:
        FlexibleConfig object with dot notation access for nested values
        
    Usage:
        python script.py --config config.yaml --model.path "new/path" --training.lr 0.001
        python script.py --config config.yaml --experiments.0.repetitions 0.2
        
        config = parse_config_with_overrides()
        model_name = config.get('model.name')
        learning_rate = config.get('training.lr', 0.001)
        repetitions = config.get('experiments.0.repetitions', 1.0)
    """
    # Create parser for config file and optional env file
    parser = argparse.ArgumentParser(description=description, add_help=False)
    parser.add_argument('--config', required=True, help='Path to YAML configuration file')
    parser.add_argument('--env', help='Path to environment file (optional)')
    parser.add_argument('--help', '-h', action='store_true', help='Show this help message and exit')
    
    # Parse known arguments to get config file path
    args, unknown = parser.parse_known_args()
    
    # Show help if requested
    if args.help:
        parser.print_help()
        print("\nAdditionally, you can override any config parameter using dot notation:")
        print("  --model.path 'new/path'       Override model.path in config")
        print("  --training.lr 0.001           Override training.lr in config")
        print("  --data.batch_size 32          Override data.batch_size in config")
        print("  --experiments.0.repetitions 0.2  Override repetitions in first experiment")
        exit(0)
    
    # Load base configuration from YAML file
    try:
        config = load_yaml_config(args.config, args.env)
        print(f"Loaded configuration from: {args.config}")
    except Exception as e:
        print(f"Error loading configuration file '{args.config}': {e}")
        exit(1)
    
    # Apply overrides directly to the loaded config
    i = 0
    while i < len(unknown):
        if unknown[i].startswith('--'):
            key = unknown[i][2:]  # Remove '--' prefix
            
            if i + 1 < len(unknown) and not unknown[i + 1].startswith('--'):
                # Parameter with value
                value = convert_value(unknown[i + 1])
                try:
                    set_nested_value_direct(config, key, value)
                    print(f"Override: {key} = {value}")
                except (ValueError, IndexError) as e:
                    print(f"Error setting {key}: {e}")
                i += 2
            else:
                # Flag parameter (boolean True)
                try:
                    set_nested_value_direct(config, key, True)
                    print(f"Override: {key} = True")
                except (ValueError, IndexError) as e:
                    print(f"Error setting {key}: {e}")
                i += 1
        else:
            print(f"Warning: Ignoring positional argument: {unknown[i]}")
            i += 1
    
    # Return as FlexibleConfig object with dot notation access
    return FlexibleConfig(config)


def parse_flexible_config(parser: argparse.ArgumentParser, override_known: bool = True):
    """
    Parse configuration using an existing argparse parser combined with YAML config and dot notation overrides.
    
    Args:
        parser: Existing argparse.ArgumentParser instance
        override_known: If True, use parse_known_args to allow unknown arguments for dot notation overrides
        
    Returns:
        Tuple of (args, config) where:
        - args: Parsed arguments from the parser
        - config: FlexibleConfig object with YAML config and dot notation overrides applied
        
    Usage:
        parser = argparse.ArgumentParser(description="My script")
        parser.add_argument("--experiment", type=str)
        parser.add_argument("--save_folder", type=str)
        args, config = parse_flexible_config(parser, override_known=True)
        
        # Access traditional args
        print(args.experiment)
        
        # Access config with dot notation
        model_name = config.get('model.name')
        repetitions = config.get('experiments.0.repetitions')
    """
    # Add config and env arguments to the existing parser
    parser.add_argument('--config', help='Path to YAML configuration file')
    parser.add_argument('--env', help='Path to environment file (optional)')
    
    # Parse arguments
    if override_known:
        args, unknown = parser.parse_known_args()
    else:
        args = parser.parse_args()
        unknown = []
    
    # Initialize config
    config = FlexibleConfig()
    
    # Load YAML config if provided
    if hasattr(args, 'config') and args.config:
        try:
            yaml_config = load_yaml_config(args.config, args.env)
            config.update(yaml_config)
            print(f"Loaded configuration from: {args.config}")
        except Exception as e:
            print(f"Error loading configuration file '{args.config}': {e}")
            exit(1)
    
    # Apply dot notation overrides directly to config from unknown arguments
    if override_known and unknown:
        i = 0
        while i < len(unknown):
            if unknown[i].startswith('--'):
                key = unknown[i][2:]  # Remove '--' prefix
                
                if i + 1 < len(unknown) and not unknown[i + 1].startswith('--'):
                    # Parameter with value
                    value = convert_value(unknown[i + 1])
                    try:
                        set_nested_value_direct(config, key, value)
                        print(f"Override: {key} = {value}")
                    except (ValueError, IndexError) as e:
                        print(f"Error setting {key}: {e}")
                    i += 2
                else:
                    # Flag parameter (boolean True)
                    try:
                        set_nested_value_direct(config, key, True)
                        print(f"Override: {key} = True")
                    except (ValueError, IndexError) as e:
                        print(f"Error setting {key}: {e}")
                    i += 1
            else:
                print(f"Warning: Ignoring positional argument: {unknown[i]}")
                i += 1
    
    return args, config


if __name__ == "__main__":
    # Example usage
    print("=== Flexible Configuration Parser with Dot Notation Overrides ===")
    
    # Method 1: Functional interface
    config = parse_config_with_overrides("Example flexible configuration parser")
    print("\nFinal configuration:")
    print(yaml.dump(dict(config), default_flow_style=False, sort_keys=True))
    
    # Demonstrate dot notation access
    print("\n=== Dot Notation Access Examples ===")
    print(f"Model name: {config.get('model.name', 'default_model')}")
    print(f"Learning rate: {config.get('training.lr', 0.001)}")
    print(f"Batch size: {config.get('data.batch_size', 32)}")
    print(f"Non-existent key: {config.get('foo.bar.baz', 'not found')}")
    
    # Demonstrate list index access
    if config.has('experiments'):
        print(f"\n=== List Index Access Examples ===")
        print(f"First experiment name: {config.get('experiments.0.name', 'not found')}")
        print(f"First experiment repetitions: {config.get('experiments.0.repetitions', 'not found')}")
    
    # Check if keys exist
    print(f"\nKey 'model.name' exists: {config.has('model.name')}")
    print(f"Key 'experiments.0.name' exists: {config.has('experiments.0.name')}")
    print(f"Key 'nonexistent.key' exists: {config.has('nonexistent.key')}")
    
    # Set new values
    config.set('runtime.experiment_id', 'exp_001')
    print(f"Set runtime.experiment_id: {config.get('runtime.experiment_id')}")
    
    # Set value in list
    if config.has('experiments'):
        config.set('experiments.0.new_param', 'test_value')
        print(f"Set experiments.0.new_param: {config.get('experiments.0.new_param')}")
    
    # Method 3: Hybrid approach with existing parser
    print("\n=== Hybrid Approach Example ===")
    example_parser = argparse.ArgumentParser(description="Continual pretraining experiment")
    example_parser.add_argument("--experiment", type=str, help="Experiment name")
    example_parser.add_argument("--save_folder", type=str, help="Save folder")
    example_parser.add_argument("--num_gpus", type=int, help="Number of GPUs")
    
    # This would be called with: python script.py --config config.yaml --experiment my_exp --experiments.0.repetitions 0.2
    # args, config = parse_flexible_config(example_parser, override_known=True)
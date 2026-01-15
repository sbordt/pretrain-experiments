"""
Inference Engine Abstraction Layer

This module provides an abstract base class for different inference backends
(vLLM, transformers AutoModel, etc.) with a unified interface.
"""

from abc import ABC, abstractmethod
from typing import Union, List, Dict, Any, Optional
import os
import yaml
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams, TokensPrompt

# Configure logger for this module
logger = logging.getLogger(__name__)

# Ensure INFO messages are visible if logging isn't already configured
# Use vLLM-style format: INFO 10-29 18:00:06 [file.py:205]
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%m-%d %H:%M:%S'
    )


def load_system_inference_defaults() -> Dict[str, Any]:
    """
    Load system-level inference defaults from a YAML file.

    Reads from the INFERENCE_DEFAULTS_PATH environment variable.
    If not set or file doesn't exist, returns an empty dict.

    Returns:
        Dictionary with backend-specific defaults and optional default_engine, e.g.:
        {
            'default_engine': 'transformers',  # Optional default engine/backend
            'vllm': {'max_num_seqs': 32, 'dtype': 'auto', ...},
            'transformers': {'max_num_seqs': 8, 'dtype': 'bfloat16', ...}
        }
        Returns empty dict if no defaults file is found.
    """
    path = os.getenv('INFERENCE_DEFAULTS_PATH')

    if not path:
        logger.info("INFERENCE_DEFAULTS_PATH not set, using empty system defaults")
        return {}

    if not os.path.exists(path):
        logger.warning(f"INFERENCE_DEFAULTS_PATH set to {path} but file does not exist")
        return {}

    try:
        logger.info(f"Loading system inference defaults from: {path}")
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
            
            # Extract default_engine and inference_defaults
            result = {}
            
            # Get the default engine if specified
            if 'default_engine' in data:
                result['default_engine'] = data['default_engine']
                logger.info(f"System default engine: {result['default_engine']}")
            
            # Get the backend-specific defaults
            inference_defaults = data.get('inference_defaults', {})
            if inference_defaults:
                # Merge inference_defaults directly into result
                result.update(inference_defaults)
                logger.info(f"Loaded system defaults for backends: {list(inference_defaults.keys())}")
            
            return result
    except Exception as e:
        logger.error(f"Failed to load inference defaults from {path}: {e}")
        return {}


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two configuration dictionaries.

    Args:
        base: Base configuration dictionary
        override: Override configuration dictionary (takes precedence)

    Returns:
        Merged configuration dictionary
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


class InferenceEngineFactory:
    """Factory class for creating inference engines."""

    # Registry of available engine types
    _ENGINE_REGISTRY = {
        'vllm': None,  # Will be set after class definitions
        'transformers': None,  # Will be set after class definitions
    }

    @classmethod
    def create(cls, engine_type: str, model: str, **kwargs) -> 'InferenceEngine':
        """
        Create an inference engine.

        Args:
            engine_type: Type of engine ('vllm' or 'transformers')
            model: Model name or path
            **kwargs: Additional parameters passed to engine constructor

        Returns:
            Configured InferenceEngine instance

        Example:
            >>> engine = InferenceEngineFactory.create('vllm', 'allenai/OLMo-2-0425-1B', max_num_seqs=4)
            >>> engine = InferenceEngineFactory.create('transformers', 'gpt2', torch_dtype=torch.bfloat16)
        """
        engine_type = engine_type.lower()

        if engine_type not in cls._ENGINE_REGISTRY:
            available = ', '.join(cls._ENGINE_REGISTRY.keys())
            raise ValueError(
                f"Unknown engine type: '{engine_type}'. "
                f"Available types: {available}"
            )

        engine_class = cls._ENGINE_REGISTRY[engine_type]

        # Create engine with model and any additional kwargs
        if engine_type == 'vllm':
            return engine_class(model=model, **kwargs)
        elif engine_type == 'transformers':
            return engine_class(model_name_or_path=model, **kwargs)
        else:
            return engine_class(model, **kwargs)

    @classmethod
    def create_from_config(
        cls,
        model: str,
        config: Optional[Union[str, Dict[str, Any]]] = None,
        **kwarg_overrides
    ) -> 'InferenceEngine':
        """
        Create an inference engine with system defaults and config-based settings.

        Priority order (highest to lowest):
        1. kwarg_overrides passed directly to this method
        2. Config argument (experiment config inference section)
        3. System default engine (from INFERENCE_DEFAULTS_PATH env var)
        4. Hardcoded fallback ('vllm')

        Args:
            model: Model name or path
            config: Either:
                - Path to YAML config file (str)
                - Dictionary with config overrides (dict)
                - None (use only system defaults)
            **kwarg_overrides: Direct overrides for any parameter (highest priority)

        Returns:
            Configured InferenceEngine instance

        Example:
            >>> # Using system defaults only
            >>> engine = InferenceEngineFactory.create_from_config('gpt2')
            >>>
            >>> # Using experiment config file
            >>> engine = InferenceEngineFactory.create_from_config(
            ...     'allenai/OLMo-2-0425-1B',
            ...     config='config/experiments/my_experiment.yaml'
            ... )
            >>>
            >>> # Using config dict
            >>> engine = InferenceEngineFactory.create_from_config(
            ...     'gpt2',
            ...     config={'inference': {'backend': 'vllm', 'backend_args': {'max_num_seqs': 16}}}
            ... )
            >>>
            >>> # With direct overrides
            >>> engine = InferenceEngineFactory.create_from_config(
            ...     'gpt2',
            ...     config='config/experiments/my_experiment.yaml',
            ...     max_num_seqs=4  # Override config value
            ... )
        """
        logger.info("Creating inference engine from config")
        logger.info(f"Model: {model}")

        # Load system defaults
        system_defaults = load_system_inference_defaults()

        # Load config if provided
        experiment_config = {}
        if config is not None:
            if isinstance(config, str):
                # Config is a file path - load it
                if os.path.exists(config):
                    logger.info(f"Loading experiment config from: {config}")
                    with open(config, 'r') as f:
                        experiment_config = yaml.safe_load(f) or {}
                else:
                    raise FileNotFoundError(f"Config file not found: {config}")
            elif isinstance(config, dict):
                # Config is already a dictionary
                logger.info("Using provided config dictionary")
                experiment_config = config
            else:
                raise TypeError(
                    f"config must be a file path (str) or dictionary (dict), "
                    f"got {type(config)}"
                )
        else:
            logger.info("No experiment config provided")

        # Extract inference section from experiment config
        inference_config = experiment_config.get('inference', {})

        # Determine backend with priority: 
        # kwarg_overrides > experiment config > system default engine > hardcoded default (vllm)
        backend = (
            kwarg_overrides.pop('backend', None) or 
            inference_config.get('backend') or 
            system_defaults.get('default_engine') or 
            'vllm'
        )

        logger.info(f"Selected backend: {backend}")

        # Get system defaults for this backend
        backend_system_defaults = system_defaults.get(backend, {})
            
        if backend_system_defaults:
            logger.info(f"System defaults for {backend}: {backend_system_defaults}")
        else:
            logger.info(f"No system defaults found for {backend}")

        # Merge backend_args: system defaults <- experiment config <- kwarg overrides
        backend_args = backend_system_defaults.copy()

        experiment_backend_args = inference_config.get('backend_args', {})
        if experiment_backend_args:
            logger.info(f"Experiment config overrides: {experiment_backend_args}")
            backend_args = merge_configs(backend_args, experiment_backend_args)

        if kwarg_overrides:
            logger.info(f"Kwarg overrides: {kwarg_overrides}")
            backend_args = merge_configs(backend_args, kwarg_overrides)

        logger.info(f"Final backend args: {backend_args}")

        # Create engine using the standard create method
        return cls.create(backend, model, **backend_args)


class InferenceEngine(ABC):
    """Abstract base class for inference engines."""

    def __init__(self):
        """Initialize inference engine with default kwargs storage."""
        # Dictionary to store default kwargs that can be used by subclasses
        self._default_kwargs = {}

    def _set_default(self, key: str, value: Any) -> None:
        """
        Set a default value for a parameter.

        Args:
            key: Parameter name
            value: Default value
        """
        self._default_kwargs[key] = value

    def _get_default(self, key: str, fallback: Any = None) -> Any:
        """
        Get a default value for a parameter.

        Args:
            key: Parameter name
            fallback: Fallback value if key not found

        Returns:
            Default value or fallback
        """
        return self._default_kwargs.get(key, fallback)

    @abstractmethod
    def generate_text(
        self,
        prompts: List[Union[str, List[int]]],
        max_tokens: int,
        temperature: float = 1.0,
        return_token_ids: bool = False,
        **kwargs
    ) -> List[Union[str, List[int]]]:
        """
        Generate text from prompts.

        Args:
            prompts: List of prompts (either strings or token ID lists)
            max_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (0.0 for deterministic, higher for more random)
            return_token_ids: If True, return token IDs instead of text
            **kwargs: Additional generation parameters (backend-specific)

        Returns:
            List of generated texts or token IDs
        """
        pass

    @abstractmethod
    def get_logprobs(
        self,
        prompts: List[Union[str, List[int]]]
    ) -> List[Dict[str, Any]]:
        """
        Get log probabilities for prompts.

        Args:
            prompts: List of prompts (either strings or token ID lists)

        Returns:
            List of dicts with 'token_ids' and 'logprobs' keys.
            'logprobs' is a list where the first element is None (no logprob for first token).
        """
        pass


class VLLMInferenceEngine(InferenceEngine):
    """Inference engine using vLLM backend."""

    def __init__(
        self,
        model: Union[str, LLM],
        dtype: str = 'auto',
        max_num_seqs: Optional[int] = None,
        **vllm_kwargs
    ):
        """
        Initialize vLLM inference engine.

        Args:
            model: Model name/path (str) or existing LLM instance
            dtype: Data type for model weights ('auto', 'float32', 'float16', 'bfloat16')
            max_num_seqs: Maximum number of sequences to process in parallel (passed to vLLM)
            **vllm_kwargs: Additional arguments passed to LLM constructor
        """
        super().__init__()

        # Store max_num_seqs for consistency with TransformersInferenceEngine
        # For vLLM, this is passed to the LLM constructor and handled internally
        self.max_num_seqs = max_num_seqs

        # Add max_num_seqs to vllm_kwargs if provided
        if max_num_seqs is not None:
            vllm_kwargs['max_num_seqs'] = max_num_seqs

        if isinstance(model, str):
            self.llm = LLM(model=model, dtype=dtype, **vllm_kwargs)
        else:
            self.llm = model

    def generate_text(
        self,
        prompts: List[Union[str, List[int]]],
        max_tokens: int,
        temperature: float = 1.0,
        return_token_ids: bool = False,
        **kwargs
    ) -> List[Union[str, List[int]]]:
        """
        Generate text using vLLM.

        Args:
            prompts: List of prompts (strings or token ID lists)
            max_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (0.0 for deterministic, higher for more random)
            return_token_ids: If True, return token IDs instead of text
            **kwargs: Additional parameters passed to SamplingParams (e.g., stop, top_p, top_k)

        Returns:
            List of generated texts or token IDs
        """
        # Convert token ID prompts to TokensPrompt format if needed
        processed_prompts = prompts
        if prompts and isinstance(prompts[0], list):
            processed_prompts = [TokensPrompt(prompt_token_ids=tokens) for tokens in prompts]

        # Create sampling parameters with standardized max_tokens and temperature
        params = SamplingParams(max_tokens=max_tokens, temperature=temperature, **kwargs)

        # Generate
        outputs = self.llm.generate(processed_prompts, sampling_params=params)

        # Extract results
        if return_token_ids:
            return [out.outputs[0].token_ids for out in outputs]
        return [out.outputs[0].text for out in outputs]

    
    def get_logprobs(
        self,
        prompts: List[Union[str, List[int]]]
    ) -> List[Dict[str, Any]]:
        """
        Get log probabilities using vLLM.

        Args:
            prompts: List of prompts (strings or token ID lists)

        Returns:
            List of dicts with 'token_ids' and 'logprobs' keys
        """
        # Sampling params for logprob extraction
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1,
            prompt_logprobs=0,
            detokenize=False,
        )

        # Convert token ID prompts to TokensPrompt format if needed
        processed_prompts = prompts
        if prompts and isinstance(prompts[0], list):
            processed_prompts = [TokensPrompt(prompt_token_ids=tokens) for tokens in prompts]

        # Generate to get logprobs
        outputs = self.llm.generate(processed_prompts, sampling_params)

        # Extract logprobs
        result = []
        for output in outputs:
            # Extract logprobs for each token (skip first as it has no logprob)
            logprobs = [None]  # First token has no logprob

            # Get the logprob for each actual token in the sequence
            for i, token_id in enumerate(output.prompt_token_ids[1:], start=1):
                # prompt_logprobs[i] is a dict mapping token_id -> Logprob object
                # We want the logprob for the specific token_id at this position
                logprobs.append(output.prompt_logprobs[i][token_id].logprob)

            result.append({
                'token_ids': output.prompt_token_ids,
                'logprobs': logprobs,
            })

        return result
    
class TransformersInferenceEngine(InferenceEngine):
    """Inference engine using transformers AutoModelForCausalLM."""

    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        trust_remote_code: bool = True,
        max_num_seqs: int = 8,
        **kwargs
    ):
        """
        Initialize transformers inference engine.

        Args:
            model_name_or_path: HuggingFace model name or path
            device: Device to load model on (e.g., 'cuda', 'cpu'). If None, auto-detect.
            dtype: Data type for model weights. Can be:
                - torch.dtype (e.g., torch.float32, torch.bfloat16)
                - str (e.g., 'float32', 'bfloat16', 'float16')
                - None (defaults to bfloat16)
            trust_remote_code: Whether to trust remote code
            max_num_seqs: Maximum number of sequences to process in parallel (default: 8)
            **model_kwargs: Additional arguments passed to AutoModelForCausalLM.from_pretrained and AutoTokenizer.from_pretrained
        """
        super().__init__()

        # Store max_num_seqs for use in generate_text and get_logprobs
        self.max_num_seqs = max_num_seqs

        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')

        # Convert dtype to torch.dtype if provided as string
        if dtype is None:
            self.torch_dtype = torch.bfloat16  # Default to bfloat16 for efficiency
        elif isinstance(dtype, str):
            dtype_map = {
                'float32': torch.float32,
                'float16': torch.float16,
                'bfloat16': torch.bfloat16,
                'float': torch.float32,
                'half': torch.float16,
            }
            self.torch_dtype = dtype_map.get(dtype.lower(), torch.bfloat16)
        else:
            self.torch_dtype = dtype

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            **kwargs
        )

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # For decoder-only models (like GPT-2), use left padding for batch generation
        # This ensures that generation starts from the actual prompt, not from padding tokens
        self.tokenizer.padding_side = 'left'

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=self.torch_dtype,
            trust_remote_code=trust_remote_code,
            **kwargs
        ).to(self.device)

        self.model.eval()

    def generate_text(
        self,
        prompts: List[Union[str, List[int]]],
        max_tokens: int,
        temperature: float = 1.0,
        return_token_ids: bool = False,
        **kwargs
    ) -> List[Union[str, List[int]]]:
        """
        Generate text using transformers with controlled batching to prevent OOM.

        Args:
            prompts: List of prompts (strings or token ID lists)
            max_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (0.0 for deterministic, higher for more random)
            return_token_ids: If True, return token IDs instead of text
            **kwargs: Additional parameters passed to model.generate (e.g., do_sample, top_p, top_k)

        Returns:
            List of generated texts or token IDs
        """
        from tqdm import tqdm

        # Process prompts in batches
        all_results = []
        num_prompts = len(prompts)

        for batch_start in tqdm(range(0, num_prompts, self.max_num_seqs)):
            batch_end = min(batch_start + self.max_num_seqs, num_prompts)
            batch_prompts = prompts[batch_start:batch_end]
            
            # Tokenize if needed
            if isinstance(batch_prompts[0], str):
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                input_lengths = [len(inputs['input_ids'][i]) for i in range(len(batch_prompts))]
            else:
                # Prompts are already token IDs
                max_len = max(len(p) for p in batch_prompts)
                padded_prompts = []
                attention_mask = []
                
                for p in batch_prompts:
                    pad_len = max_len - len(p)
                    padded = [self.tokenizer.pad_token_id] * pad_len + p
                    mask = [0] * pad_len + [1] * len(p)
                    padded_prompts.append(padded)
                    attention_mask.append(mask)
                
                inputs = {
                    'input_ids': torch.tensor(padded_prompts, device=self.device),
                    'attention_mask': torch.tensor(attention_mask, device=self.device)
                }
                input_lengths = [len(padded_prompts[i]) for i in range(len(batch_prompts))]
            
            # Determine sampling mode
            do_sample = temperature > 0
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if do_sample else None,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **kwargs
                )
            
            # Extract generated tokens
            batch_results = []
            for i, output in enumerate(outputs):
                input_len = input_lengths[i]
                generated_ids = output[input_len:].tolist()
                
                if return_token_ids:
                    batch_results.append(generated_ids)
                else:
                    batch_results.append(self.tokenizer.decode(generated_ids, skip_special_tokens=True))
            
            all_results.extend(batch_results)
            
            # Clear cache between batches to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return all_results

    def get_logprobs(
        self,
        prompts: List[Union[str, List[int]]]
    ) -> List[Dict[str, Any]]:
        """
        Get log probabilities using transformers with efficient batching.

        Args:
            prompts: List of prompts (strings or token ID lists)

        Returns:
            List of dicts with 'token_ids' and 'logprobs' keys
        """
        results = [None] * len(prompts)  # Pre-allocate results list

        # Process prompts in batches
        for batch_start in range(0, len(prompts), self.max_num_seqs):
            batch_end = min(batch_start + self.max_num_seqs, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]
            
            # Prepare batch inputs
            batch_token_ids = []
            
            for prompt in batch_prompts:
                if isinstance(prompt, str):
                    tokens = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True
                    )
                    token_ids = tokens['input_ids'][0].tolist()
                else:
                    token_ids = prompt
                
                batch_token_ids.append(token_ids)
            
            # Pad sequences to same length for batching
            max_len = max(len(ids) for ids in batch_token_ids)
            padded_token_ids = []
            attention_masks = []
            
            for token_ids in batch_token_ids:
                # Left padding for decoder-only models
                pad_len = max_len - len(token_ids)
                padded_ids = [self.tokenizer.pad_token_id] * pad_len + token_ids
                mask = [0] * pad_len + [1] * len(token_ids)
                
                padded_token_ids.append(padded_ids)
                attention_masks.append(mask)
            
            # Create batch tensors
            inputs = {
                'input_ids': torch.tensor(padded_token_ids, device=self.device),
                'attention_mask': torch.tensor(attention_masks, device=self.device)
            }
            
            # Forward pass for entire batch
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)
            
            # Compute log probabilities
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            
            # Extract logprobs for each sequence in the batch
            for i, (original_token_ids, padded_ids) in enumerate(zip(batch_token_ids, padded_token_ids)):
                # Find where actual tokens start (after padding)
                pad_len = max_len - len(original_token_ids)
                
                # Get logprobs for each token (first token has no logprob)
                token_logprobs = [None]  # First token has no logprob
                
                for j, token_id in enumerate(original_token_ids[1:], start=1):
                    # Adjust index for padding
                    padded_j = j + pad_len
                    # log_probs[i, padded_j-1] contains logprobs for the position after token padded_j-1
                    logprob = log_probs[i, padded_j - 1, token_id].item()
                    token_logprobs.append(logprob)
                
                # Store result in the correct position
                results[batch_start + i] = {
                    'token_ids': original_token_ids,
                    'logprobs': token_logprobs,
                }
        
        return results


# Register engine classes in the factory
InferenceEngineFactory._ENGINE_REGISTRY['vllm'] = VLLMInferenceEngine
InferenceEngineFactory._ENGINE_REGISTRY['transformers'] = TransformersInferenceEngine



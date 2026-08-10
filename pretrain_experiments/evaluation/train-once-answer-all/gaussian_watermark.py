import torch
import pickle
import argparse
import os
import re
import sys
from typing import TypeVar
from torch.nn import CrossEntropyLoss
from torch.distributions.normal import Normal
from transformers import AutoTokenizer, AutoModelForCausalLM

# Define a generic TypeVar for type hinting
T = TypeVar('T')

def move_to_device(o: T, device: torch.device) -> T:
    """Recursively moves tensors in a nested structure to the specified device."""
    if isinstance(o, torch.Tensor):
        return o.to(device)
    elif isinstance(o, dict):
        return {k: move_to_device(v, device) for k, v in o.items()}
    elif isinstance(o, list):
        return [move_to_device(x, device) for x in o]
    elif isinstance(o, tuple):
        return tuple((move_to_device(x, device) for x in o))
    else:
        return o

def compute_causal_loss(pred_scores:torch.tensor, labels:torch.tensor) -> torch.tensor:
    """ Computes the causal language modeling loss for a given set of prediction scores and labels.
    Args:
        pred_scores (torch.tensor): The prediction scores from the model, shape: (batch_size, sequence_length, vocab_size).
        labels (torch.tensor): The ground truth labels, shape: (batch_size, sequence_length).
    Returns:
        torch.tensor: The computed loss value.
    """
    # We are doing next-token prediction; shift prediction scores and input ids by one
    shift_logits = pred_scores[:, :-1, :].contiguous()
    labels = labels[:, 1:].contiguous()
    loss_fct = CrossEntropyLoss()
    lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
    return lm_loss

def _compute_dot(gradient: torch.tensor, noise: torch.tensor, n_std: float) -> torch.tensor:
    """
    Computes the dot product between gradients and Gaussian noise, normalized by the standard deviation.
    
    Args:
        gradient: input gradients of loss with respect to input embeddings, shape: (batch_size, embedding_dim)
        noise: Gaussian noise added to the input embeddings, shape: (batch_size, embedding_dim)
        n_std: standard deviation of the Gaussian noise, a scalar
    Returns:
        torch.tensor: dot products of the gradients and noise, shape: (batch_size,)
    """
    # Use double precision for accuracy
    gradient = gradient.to(torch.float32)
    noise = noise.to(torch.float32)

    dot_prods = (gradient * noise) / n_std
    dot_prods = torch.sum(dot_prods, dim=1)
    dot_prods = dot_prods / torch.norm(gradient, 2, dim=1)
    dot_prods = dot_prods.detach()
    return dot_prods


def gaussian_privacy_score(noise_dir: str, model_dir:str, revision: str, cache_dir: str, results_dir: str, noise_std: float) -> dict:
    """
    Calculates the privacy score based on the dot product of gradients and injected noise.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load noise data from .pkl file
    print(f"Loading noise data from: {noise_dir}")
    with open(noise_dir, 'rb') as f:
        noise_data = pickle.load(f)

    # Load model and tokenizer
    print(f"Loading model and tokenizer from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        "allenai/OLMo-2-0425-1B",
        use_fast=False, 
        trust_remote_code=True, 
        cache_dir=cache_dir
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16, # to save memory
        revision=revision if revision is not None else "main", 
        trust_remote_code=True, 
        cache_dir=cache_dir
    ).to(device)
    model.gradient_checkpointing_enable() # to save memory
    print("Model and tokenizer loaded successfully.")

    dots = []
    dots_test = []
    counter = 0

    # check dimensions of noise_data
    print(f"Number of items in noise data: {len(noise_data)}")
    print(f"Example item shape: {noise_data[0][-1].shape if len(noise_data[0]) > 2 else 'N/A'}")
    # memory required for processing
    print(f"Estimated memory required for processing file: {len(noise_data) * noise_data[0][-1].element_size() * noise_data[0][-1].nelement() / (1024**3):.2f} GB")

    with torch.enable_grad():
        for item in noise_data:
            counter += 1
            # if counter % 10 == 0:
            print(f"Processing batch {counter}...")
            
            # Handle different pickle file formats
            if len(item) == 4:
                # Format: (batch_id, _, token_ids, noise)
                batch_id, _, token_ids, noise = item
            elif len(item) == 3:
                # Format: (token_ids, sequence_seed, noise)
                token_ids, sequence_seed, noise = item
            else:
                print(f"Unexpected item format with {len(item)} elements. Skipping...")
                continue

            # Add a batch dimension of 1 to the tensors
            token_ids = token_ids.unsqueeze(0)
            noise = noise.unsqueeze(0)

            # --- Hotfix ---
            if "13B" in model_dir:
                MAX_LEN = 2048 
                if token_ids.shape[1] > MAX_LEN:
                    token_ids = token_ids[:, :MAX_LEN]
                    noise = noise[:, :MAX_LEN]
            # --- END Hotfix ---

            batch = {
                'input_ids': token_ids,
                'attention_mask': torch.ones_like(token_ids),
                'labels': token_ids,
            }
            batch = move_to_device(batch, device)
            noise = move_to_device(noise, device)
            
            # Get input embeddings for the current batch
            inputs_embeds = model.get_input_embeddings()(batch['input_ids'])
            # print("No OOM so far - stage 1a...")
            # print(f"inputs_embeds shape: {inputs_embeds.shape}, noise shape: {noise.shape}")
            inputs_embeds.requires_grad_(True)

            # Generate fresh test noise for sanity check
            test_noise = torch.randn_like(inputs_embeds) * noise_std
            # print("No OOM so far - stage 1b...")

            # Add Memory check
            # print("\n" + "="*80)
            # print(f"--- RUNNING NVIDIA-SMI (First Item, SeqLen={inputs_embeds.shape[1]}) ---")
            # os.system('nvidia-smi')
            # print("="*80 + "\n")

            # Forward pass and loss calculation
            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=batch['attention_mask']
            )
            logits = outputs.logits
            
            lm_loss = compute_causal_loss(logits, batch['labels'])

            # Compute gradients w.r.t. embeddings
            # print("No OOM so far - stage 2...")
            grads = torch.autograd.grad(lm_loss, inputs_embeds)[0].detach()
            
            # Compute dot products
            dot = _compute_dot(grads.flatten(1), noise.flatten(1), noise_std)
            # print("No OOM so far - stage 3...")
            dot_test = _compute_dot(grads.flatten(1), test_noise.flatten(1), noise_std)
            # print(f"Dot product (test noise): {dot_test}")

            # Store results
            dots.append(dot.cpu())
            dots_test.append(dot_test.cpu())
            
            # Clean up to free memory
            del grads, lm_loss, logits, outputs, inputs_embeds, test_noise, dot, dot_test
            # Force PyTorch to empty its VRAM cache
            torch.cuda.empty_cache()

    # Concatenate results from all batches
    all_dots = torch.cat(dots)
    all_dots_test = torch.cat(dots_test)

    # Compute p-values
    n_samples = len(all_dots)
    if n_samples < 2:
        print("Not enough samples to compute p-value.")
        return

    n_dist = Normal(0, 1)  # Standard normal distribution for p-value calculation
    
    # --- In-distribution (using pre-generated noise) ---
    mean_in = torch.mean(all_dots)
    std_in = torch.std(all_dots)
    sem_in = std_in / torch.sqrt(torch.tensor(n_samples, dtype=torch.float32))
    print(f"\n--- Results (In-Distribution Noise) ---")
    print(f"Mean: {mean_in.item():.6f}, Std: {std_in.item():.6f}")

    t_statistic_in = (mean_in - 0) / sem_in if sem_in > 0 else 0.0
    p_value_in = 2 * (1 - n_dist.cdf(torch.abs(t_statistic_in)))

    # --- Out-of-distribution (using fresh test noise) ---
    mean_out = torch.mean(all_dots_test)
    std_out = torch.std(all_dots_test)
    sem_out = std_out / torch.sqrt(torch.tensor(n_samples, dtype=torch.float32))
    print(f"\n--- Sanity Check (Fresh Gaussian Noise) ---")
    print(f"Mean: {mean_out.item():.6f}, Std: {std_out.item():.6f}")
    
    t_statistic_out = (mean_out - 0) / sem_out if sem_out > 0 else 0.0
    p_value_out = 2 * (1 - n_dist.cdf(torch.abs(t_statistic_out)))
    
    print("\n--- P-Values ---")
    print(f"P-Value (In-Distribution): {p_value_in.item():.6f}")
    print(f"P-Value (Out-of-Distribution/Sanity Check): {p_value_out.item():.6f}")

    return all_dots, all_dots_test


def find_gaussian_poisoning_files(noise_dir: str) -> list:
    """
    Find all files in the directory that match the Gaussian poisoning pattern.
    
    Args:
        noise_dir: Directory containing the noise files
        
    Returns:
        list: Sorted list of tuples (step_number, file_path)
    """
    if not os.path.isdir(noise_dir):
        raise ValueError(f"Directory {noise_dir} does not exist")
    
    # Pattern to match various formats:
    # - gaussian_poisoning_seeds_and_sequences_sampled_<number>.pkl
    # - gaussian_poisoning_seeds_and_sequences_<number>.pkl
    # - gaussian_poisoning_seeds_and_sequences_step=<number>.pkl
    pattern = r'gaussian_poisoning_seeds_and_sequences(?:_sampled)?(?:_(\d+)|_step=(\d+))\.pkl'
    
    matching_files = []
    
    for filename in os.listdir(noise_dir):
        match = re.match(pattern, filename)
        if match:
            # Either group 1 or group 2 will contain the step number
            step_number = int(match.group(1) if match.group(1) else match.group(2))
            file_path = os.path.join(noise_dir, filename)
            matching_files.append((step_number, file_path))
    
    # Sort by step number
    matching_files.sort(key=lambda x: x[0])
    
    return matching_files


def aggregate_scores(args):
    """
    Aggregates privacy scores from multiple files
    """
    # Find all matching files
    matching_files = find_gaussian_poisoning_files(args.noise_dir)
    
    if not matching_files:
        print(f"No matching Gaussian poisoning files found in {args.noise_dir}")
        print("Expected pattern: gaussian_poisoning_seeds_and_sequences_sampled_<number>.pkl")
        return
    
    print(f"Found {len(matching_files)} matching files:")
    for step_num, file_path in matching_files:
        print(f"  Step {step_num}: {os.path.basename(file_path)}")
    
    all_scores_in = []
    all_scores_out = []

    for i, (step_num, noise_file) in enumerate(matching_files, 1):
        print(f"\n--- Processing file {i}/{len(matching_files)}: Step {step_num} ---")
        print(f"Noise file: {noise_file}")

        try:
            dots_in, dots_out = gaussian_privacy_score(
                noise_dir=noise_file,
                model_dir=args.model_dir,
                revision=args.revision,
                cache_dir=args.cache_dir,
                noise_std=args.noise_std,
                results_dir=args.results_dir
            )
            all_scores_in.append(dots_in.cpu())
            all_scores_out.append(dots_out)
        except Exception as e:
            print(f"Error processing {noise_file}: {e}")
            continue

    if not all_scores_in:
        print("No files were successfully processed.", file=sys.stderr)
        sys.exit(2)

    # Concatenate results from all batches
    all_dots = torch.cat(all_scores_in)
    all_dots_test = torch.cat(all_scores_out)
    
    mean_in = torch.mean(all_dots)
    std_in = torch.std(all_dots)
    n_samples = len(all_dots)
    sem_in = std_in / torch.sqrt(torch.tensor(n_samples, dtype=torch.float32))
    print(f"\n--- Aggregated Results (In-Distribution Noise) ---")
    print(f"Mean: {mean_in.item():.6f}, Std: {std_in.item():.6f}")
    print(f"Total samples: {n_samples}")

    mean_out = torch.mean(all_dots_test)
    std_out = torch.std(all_dots_test)
    n_samples_out = len(all_dots_test)
    sem_out = std_out / torch.sqrt(torch.tensor(n_samples_out, dtype=torch.float32))
    print(f"\n--- Aggregated Sanity Check (Fresh Gaussian Noise) ---")
    print(f"Mean: {mean_out.item():.6f}, Std: {std_out.item():.6f}")
    print(f"Total samples: {n_samples_out}")

    # Extract step identifier from revision for saving
    if args.revision:
        parts = args.revision.split('-')
        if len(parts) > 1:
            n_steps = parts[1]
        else:
            n_steps = args.revision
    else:
        n_steps = "unknown"

    # Save results
    os.makedirs(args.results_dir, exist_ok=True)
    in_file = f"{args.results_dir}/gaussian_privacy_scores_in_{n_steps}.pt"
    out_file = f"{args.results_dir}/gaussian_privacy_scores_out_{n_steps}.pt"

    torch.save(all_dots, in_file)
    torch.save(all_dots_test, out_file)
    
    print(f"\nSaved results to:")
    print(f"  In-distribution scores: {in_file}")
    print(f"  Out-of-distribution scores: {out_file}")

    return all_dots, all_dots_test


def main():
    """
    Main function to parse command-line arguments and run the privacy score calculation.
    Saves raw scores to specified directory.
    """
    parser = argparse.ArgumentParser(
        description="Compute a Gaussian privacy score for a language model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--noise_dir", 
        type=str, 
        help="Path to the directory containing noise data files (.pkl), which were generated during training."
    )

    parser.add_argument(
        "--revision", 
        type=str, 
        default=None, 
        help="Revision tag for the model."
    )

    parser.add_argument(
        "--model_dir", 
        type=str, 
        help="Path or Hugging Face name of the pre-trained model."
    )
    parser.add_argument(
        "--noise_std", 
        type=float,
        default=0.075,
        help="Standard deviation of the Gaussian noise used during training. For all models this was set to 0.075."
    )
    parser.add_argument(
        "--cache_dir", 
        type=str, 
        default=None, 
        help="Optional directory to cache downloaded models."
    )

    parser.add_argument(
        "--results_dir", 
        type=str, 
        default=None, 
        help="Directory to save results to."
    )
    
    args = parser.parse_args()

    aggregate_scores(args)
    

if __name__ == "__main__":
    main()
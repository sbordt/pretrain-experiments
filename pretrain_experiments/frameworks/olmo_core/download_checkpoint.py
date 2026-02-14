# https://github.com/allenai/OLMo-core/issues/485 (changed shard discovery / download)
import argparse
import csv
import os
from pathlib import Path
import requests
from tqdm import tqdm
from urllib.parse import urljoin

def download_file(url, save_path, chunk_size=8192):
    """Download a file, returns True if successful, False if empty/skipped."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    
    # Skip empty files
    if total_size == 0:
        return False
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=save_path.name) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    return True

def get_content_length(url):
    """Get Content-Length for a URL. Returns:
       >0: file exists with content
        0: file exists but empty
       -1: file doesn't exist
    """
    try:
        response = requests.head(url)
        response.raise_for_status()
        return int(response.headers.get('content-length', 0))
    except requests.exceptions.RequestException:
        return -1

def try_get_directory_listing(url):
   # New OLMo-core format
   new_format_files = [
       "config.yaml",
       "train.pt",
       "model.safetensors",
       "optim.safetensors",
   ]
   # Old trainer format
   old_format_files = [
       ".metadata.json",
       "config.json",
   ]
   old_format_dirs = [
       "model_and_optim",
       "train",
   ]

   found_files = []
   found_dirs = []

   # Try new format first
   for pattern in new_format_files:
       try:
           test_url = urljoin(url.rstrip('/') + '/', pattern)
           response = requests.head(test_url)
           response.raise_for_status()
           found_files.append(pattern)
       except requests.exceptions.HTTPError as e:
           if response.status_code != 404:
               raise
       except requests.exceptions.RequestException:
           raise

   if found_files:
       return found_files, []

   # Try old format
   for pattern in old_format_files:
       try:
           test_url = urljoin(url.rstrip('/') + '/', pattern)
           response = requests.head(test_url)
           response.raise_for_status()
           found_files.append(pattern)
       except requests.exceptions.HTTPError as e:
           if response.status_code != 404:
               raise
       except requests.exceptions.RequestException:
           raise

   # Check for old format directories
   for dir_name in old_format_dirs:
       try:
           # Check if directory exists by probing known files
           if dir_name == "model_and_optim":
               test_url = urljoin(url.rstrip('/') + '/', f"{dir_name}/.metadata")
           else:  # train
               test_url = urljoin(url.rstrip('/') + '/', f"{dir_name}/rank0.pt")
           response = requests.head(test_url)
           response.raise_for_status()
           found_dirs.append(dir_name)
       except requests.exceptions.HTTPError as e:
           if response.status_code != 404:
               raise
       except requests.exceptions.RequestException:
           raise

   if len(found_files) <= 0 and len(found_dirs) <= 0:
       raise ValueError(f"No checkpoint files found at {url}")

   return found_files, found_dirs

def discover_directory_files(url, dir_name):
    """Discover files in a directory by probing known patterns."""
    files = []
    base_url = urljoin(url.rstrip('/') + '/', dir_name + '/')

    if dir_name == "model_and_optim":
        # Check for .metadata file
        try:
            test_url = urljoin(base_url, ".metadata")
            response = requests.head(test_url)
            response.raise_for_status()
            files.append(".metadata")
        except requests.exceptions.RequestException:
            pass

        # Step 1: Discover number of ranks by probing shard 0
        # Only count ranks that have non-zero content
        print("  Discovering ranks (probing shard 0)...")
        ranks_with_data = []
        for rank in range(512):
            filename = f"__0_{rank}.distcp"
            test_url = urljoin(base_url, filename)
            content_length = get_content_length(test_url)
            if content_length < 0:
                break  # File doesn't exist, no more ranks
            if content_length > 0:
                ranks_with_data.append(rank)
        
        if not ranks_with_data:
            print("  Warning: No non-empty rank files found for shard 0")
            return files
        
        print(f"  Found {len(ranks_with_data)} ranks with data: {ranks_with_data}")
        
        # Step 2: Discover shards by probing rank 0 (or first rank with data)
        # Only count shards that have non-zero content
        probe_rank = ranks_with_data[0]
        print(f"  Discovering shards (probing rank {probe_rank})...")
        shards_with_data = []
        consecutive_missing = 0
        for shard in range(1024):
            filename = f"__{shard}_{probe_rank}.distcp"
            test_url = urljoin(base_url, filename)
            content_length = get_content_length(test_url)
            if content_length < 0:
                consecutive_missing += 1
                # Allow some gaps, but stop if too many consecutive missing
                if consecutive_missing > 10:
                    break
                continue
            consecutive_missing = 0
            if content_length > 0:
                shards_with_data.append(shard)
        
        print(f"  Found {len(shards_with_data)} shards with data")
        
        # Step 3: Generate all filenames for shards and ranks with data
        for shard in shards_with_data:
            for rank in ranks_with_data:
                files.append(f"__{shard}_{rank}.distcp")
        
        print(f"  Total: {len(files)} checkpoint files to download")
        
    elif dir_name == "train":
        # Probe for rank files: rank0.pt, rank1.pt, etc.
        for i in range(256):
            filename = f"rank{i}.pt"
            test_url = urljoin(base_url, filename)
            content_length = get_content_length(test_url)
            if content_length < 0:
                break  # File doesn't exist
            if content_length > 0:
                files.append(filename)

    return files

def download_checkpoint(url, save_dir):
   base_path = Path(save_dir)
   base_path.mkdir(parents=True, exist_ok=True)
   print(f"Saving to: {base_path}")
   available_files, available_dirs = try_get_directory_listing(url)

   if not available_files and not available_dirs:
       raise ValueError("Matching files not found in directory")

   failed_files = []
   skipped_files = []
   downloaded_files = []

   # Download top-level files
   for file in available_files:
       file_url = urljoin(url.rstrip('/') + '/', file)
       file_path = base_path / file
       try:
           print(f"\nDownloading: {file}")
           if download_file(file_url, file_path):
               downloaded_files.append(file)
           else:
               skipped_files.append(file)
               print(f"Skipped {file} (empty)")
       except requests.exceptions.Timeout:
           print(f"Timeout error for {file}, retrying...")
           try:
               if download_file(file_url, file_path):
                   downloaded_files.append(file)
               else:
                   skipped_files.append(file)
           except requests.exceptions.RequestException as e:
               failed_files.append(file)
               print(f"Failed to download {file}: {e}")
       except requests.exceptions.RequestException as e:
           failed_files.append(file)
           print(f"Failed to download {file}: {e}")

   # Download directories (old format)
   for dir_name in available_dirs:
       print(f"\nDiscovering files in {dir_name}/...")
       dir_files = discover_directory_files(url, dir_name)
       print(f"Will download {len(dir_files)} files from {dir_name}/")

       dir_path = base_path / dir_name
       dir_path.mkdir(parents=True, exist_ok=True)

       for file in dir_files:
           file_url = urljoin(url.rstrip('/') + '/', f"{dir_name}/{file}")
           file_path = dir_path / file
           
           # Skip if already exists with non-zero size
           if file_path.exists() and file_path.stat().st_size > 0:
               print(f"Skipping {dir_name}/{file} (already exists)")
               continue
           
           try:
               print(f"\nDownloading: {dir_name}/{file}")
               if download_file(file_url, file_path):
                   downloaded_files.append(f"{dir_name}/{file}")
               else:
                   skipped_files.append(f"{dir_name}/{file}")
                   print(f"Skipped (empty)")
           except requests.exceptions.Timeout:
               print(f"Timeout error for {dir_name}/{file}, retrying...")
               try:
                   if download_file(file_url, file_path):
                       downloaded_files.append(f"{dir_name}/{file}")
                   else:
                       skipped_files.append(f"{dir_name}/{file}")
               except requests.exceptions.RequestException as e:
                   failed_files.append(f"{dir_name}/{file}")
                   print(f"Failed to download {dir_name}/{file}: {e}")
           except requests.exceptions.RequestException as e:
               failed_files.append(f"{dir_name}/{file}")
               print(f"Failed to download {dir_name}/{file}: {e}")

   print(f"\n=== Summary ===")
   print(f"Downloaded: {len(downloaded_files)} files")
   if skipped_files:
       print(f"Skipped (empty): {len(skipped_files)} files")
   if failed_files:
       print(f"FAILED: {len(failed_files)} files")
       print(f"Failed files: {failed_files}")

def main():
    parser = argparse.ArgumentParser(description='Download OLMo checkpoints')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    download_parser = subparsers.add_parser('download', 
                                          help='Download checkpoints from CSV file')
    download_parser.add_argument('csv_file', type=str,
                               help='Path to the CSV file containing checkpoint URLs')
    download_parser.add_argument('--step', type=str, required=True,
                               help='Specific step number to download')
    download_parser.add_argument('--save-dir', type=str, default='./checkpoints',
                               help='Base directory to save downloaded checkpoints')
    list_parser = subparsers.add_parser('list',
                                       help='List available checkpoint steps')
    list_parser.add_argument('csv_file', type=str,
                            help='Path to the CSV file containing checkpoint URLs')
    args = parser.parse_args()
    
    print(f"Reading CSV file: {args.csv_file}")
    
    with open(args.csv_file, 'r') as f:
        reader = csv.DictReader(f)
        urls = [(row['Step'], row['Checkpoint Directory']) for row in reader]
    
    if args.command == 'list':
        print("Available steps:")
        for step, _ in urls:
            print(f"Step {step}")
        return
    elif args.step:
        urls = [(step, url) for step, url in urls if step == args.step]
        if not urls:
            print(f"Error: Step {args.step} not found in the CSV file.")
            print("Use list argument to see available checkpoint steps.")
            return
    
    print(f"Saving checkpoints to: {args.save_dir}")
    for step, url in urls:
        print(f"\nStep {step}:")
        print(f"URL: {url}")
        save_path = os.path.join(args.save_dir, f"step{step}")
        download_checkpoint(url, save_path)
    

if __name__ == "__main__":
    main()
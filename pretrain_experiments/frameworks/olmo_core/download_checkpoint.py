# https://github.com/allenai/OLMo-core/issues/485
import argparse
import csv
import os
from pathlib import Path
import requests
from tqdm import tqdm
from urllib.parse import urljoin

def download_file(url, save_path, chunk_size=8192):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=save_path.name) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

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

        # Probe for sharded files: __0_0.distcp, __0_1.distcp, ..., __1_0.distcp, etc.
        # Need to discover both the number of shards and the number of ranks
        for shard in range(512):  # Probe up to 512 shards
            found_any_rank = False
            for rank in range(256):  # Probe up to 256 ranks per shard
                try:
                    filename = f"__{shard}_{rank}.distcp"
                    test_url = urljoin(base_url, filename)
                    response = requests.head(test_url)
                    response.raise_for_status()
                    files.append(filename)
                    found_any_rank = True
                except requests.exceptions.RequestException:
                    break  # No more ranks for this shard
            if not found_any_rank:
                break  # No more shards
        
        print(f"Discovered {len(files)} files in model_and_optim/")
        
    elif dir_name == "train":
        # Probe for rank files: rank0.pt, rank1.pt, etc.
        for i in range(256):  # Probe up to 256 ranks
            try:
                filename = f"rank{i}.pt"
                test_url = urljoin(base_url, filename)
                response = requests.head(test_url)
                response.raise_for_status()
                files.append(filename)
            except requests.exceptions.RequestException:
                break

    return files

def download_checkpoint(url, save_dir):
   base_path = Path(save_dir)
   base_path.mkdir(parents=True, exist_ok=True)
   print(f"Saving to: {base_path}")
   available_files, available_dirs = try_get_directory_listing(url)

   if not available_files and not available_dirs:
       raise ValueError("Matching files not found in directory")

   failed_files = []

   # Download top-level files
   for file in available_files:
       file_url = urljoin(url.rstrip('/') + '/', file)
       file_path = base_path / file
       try:
           print(f"\nDownloading: {file}")
           download_file(file_url, file_path)
       except requests.exceptions.Timeout:
           print(f"Timeout error for {file}, retrying...")
           try:
               download_file(file_url, file_path)
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
       print(f"Found {len(dir_files)} files in {dir_name}/")

       dir_path = base_path / dir_name
       dir_path.mkdir(parents=True, exist_ok=True)

       for file in dir_files:
           file_url = urljoin(url.rstrip('/') + '/', f"{dir_name}/{file}")
           file_path = dir_path / file
           try:
               print(f"\nDownloading: {dir_name}/{file}")
               download_file(file_url, file_path)
           except requests.exceptions.Timeout:
               print(f"Timeout error for {dir_name}/{file}, retrying...")
               try:
                   download_file(file_url, file_path)
               except requests.exceptions.RequestException as e:
                   failed_files.append(f"{dir_name}/{file}")
                   print(f"Failed to download {dir_name}/{file}: {e}")
           except requests.exceptions.RequestException as e:
               failed_files.append(f"{dir_name}/{file}")
               print(f"Failed to download {dir_name}/{file}: {e}")

   if failed_files:
       print(f"\nFAILED to download these files: {failed_files}")

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
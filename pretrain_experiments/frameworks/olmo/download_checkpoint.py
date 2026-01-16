import os
import requests
from tqdm import tqdm
from requests.exceptions import RequestException, ChunkedEncodingError, ConnectionError, Timeout
import time

def check_file_exists_on_server(url, timeout=10):
    """Check if a file exists on the server using HEAD request."""
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        return response.status_code == 200
    except (RequestException, Timeout):
        # Try GET with stream=True and close immediately as fallback
        try:
            response = requests.get(url, stream=True, timeout=timeout)
            response.close()
            return response.status_code == 200
        except:
            return False

def get_remote_file_info(url, timeout=10):
    """Get the size and support for range requests of a remote file."""
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        if response.status_code == 200:
            size = int(response.headers.get('content-length', 0))
            accepts_ranges = response.headers.get('accept-ranges', '').lower() == 'bytes'
            etag = response.headers.get('etag', '')
            return size, accepts_ranges, etag
    except:
        pass
    return 0, False, ''

def file_exists_with_correct_size(filepath, expected_size):
    """Check if file exists and has the expected size."""
    if not os.path.exists(filepath):
        return False
    
    if expected_size == 0:  # If we can't determine remote size
        return False
        
    actual_size = os.path.getsize(filepath)
    return actual_size == expected_size

def download_with_resume(url, filepath, chunk_size=1024*1024, timeout=30, max_retries_per_chunk=15):
    """Download a file with resume capability for large files."""
    headers = {}
    mode = 'wb'
    resume_pos = 0
    
    # Check if partial file exists
    if os.path.exists(filepath):
        resume_pos = os.path.getsize(filepath)
        headers['Range'] = f'bytes={resume_pos}-'
        mode = 'ab'
        print(f"Resuming download from byte {resume_pos}")
    
    response = None
    for attempt in range(max_retries_per_chunk):
        try:
            response = requests.get(url, stream=True, headers=headers, timeout=timeout)
            if response.status_code in [200, 206]:  # 206 is partial content
                break
            elif response.status_code == 416:  # Range not satisfiable - file might be complete
                return True
            else:
                print(f"Unexpected status code: {response.status_code}")
                if attempt < max_retries_per_chunk - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        except (Timeout, ConnectionError) as e:
            print(f"Connection error (attempt {attempt + 1}): {e}")
            if attempt < max_retries_per_chunk - 1:
                time.sleep(2 ** attempt)
            else:
                raise
    
    if response is None or response.status_code not in [200, 206]:
        return False
    
    # Get total size
    if response.status_code == 206:
        content_range = response.headers.get('content-range', '')
        if content_range:
            total_size = int(content_range.split('/')[-1])
        else:
            total_size = int(response.headers.get('content-length', 0)) + resume_pos
    else:
        total_size = int(response.headers.get('content-length', 0))
    
    # Download with progress bar
    progress = tqdm(
        total=total_size, 
        initial=resume_pos,
        unit='iB', 
        unit_scale=True, 
        desc=os.path.basename(filepath)
    )
    
    try:
        with open(filepath, mode) as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    progress.update(len(chunk))
    finally:
        progress.close()
    
    return True

def download_file_chunked(url, filepath, total_size, chunk_size=50*1024*1024, max_retries=15):
    """Download a large file in chunks using Range requests."""
    temp_parts = []
    part_size = 500 * 1024 * 1024  # 500MB parts
    num_parts = (total_size + part_size - 1) // part_size
    
    print(f"Downloading {num_parts} parts of ~{part_size/(1024*1024):.0f}MB each")
    
    for i in range(num_parts):
        start = i * part_size
        end = min(start + part_size - 1, total_size - 1)
        part_filepath = f"{filepath}.part{i}"
        temp_parts.append(part_filepath)
        
        if os.path.exists(part_filepath):
            part_size_on_disk = os.path.getsize(part_filepath)
            expected_part_size = end - start + 1
            if part_size_on_disk == expected_part_size:
                print(f"Part {i+1}/{num_parts} already complete")
                continue
        
        headers = {'Range': f'bytes={start}-{end}'}
        
        success = False
        for attempt in range(max_retries):
            try:
                print(f"Downloading part {i+1}/{num_parts} (bytes {start}-{end})")
                response = requests.get(url, headers=headers, stream=True, timeout=60)
                
                if response.status_code in [200, 206]:
                    with open(part_filepath, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                    success = True
                    break
                else:
                    print(f"Failed with status {response.status_code}")
                    
            except (RequestException, ConnectionError, ChunkedEncodingError) as e:
                print(f"Error downloading part {i+1} (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))  # Increasing delay
        
        if not success:
            print(f"Failed to download part {i+1} after {max_retries} attempts")
            return False
    
    # Combine parts
    print("Combining parts...")
    with open(filepath, 'wb') as outfile:
        for part_filepath in temp_parts:
            with open(part_filepath, 'rb') as infile:
                while True:
                    chunk = infile.read(chunk_size)
                    if not chunk:
                        break
                    outfile.write(chunk)
            os.remove(part_filepath)  # Clean up part file
    
    print("Download complete!")
    return True

def download_file(url, directory, filename, chunk_size=1024*1024, max_retries=3, use_chunked_for_large_files=True):
    """Download a single file with enhanced robustness for large files."""
    # Ensure directory exists
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    
    # First check if file exists on server
    print(f"Checking if {filename} exists on server...")
    if not check_file_exists_on_server(url):
        print(f"File {filename} does not exist on server (404). Skipping.")
        return
    
    # Get remote file info
    remote_size, accepts_ranges, etag = get_remote_file_info(url)
    
    # Check if file already exists with correct size
    if file_exists_with_correct_size(filepath, remote_size):
        print(f"File {filename} already exists with correct size ({remote_size:,} bytes). Skipping download.")
        return
    
    # Determine download strategy based on file size
    size_gb = remote_size / (1024**3) if remote_size > 0 else 0
    
    if use_chunked_for_large_files and remote_size > 1024**3 and accepts_ranges:  # > 1GB and supports ranges
        print(f"File size: {size_gb:.2f}GB - using chunked download with resume support")
        success = download_file_chunked(url, filepath, remote_size, chunk_size)
    else:
        # Try regular download with resume capability
        for attempt in range(max_retries):
            try:
                print(f"Downloading {filename}... (attempt {attempt + 1}/{max_retries})")
                if remote_size > 0:
                    print(f"File size: {size_gb:.2f}GB")
                
                if accepts_ranges and remote_size > 100*1024*1024:  # > 100MB
                    # Use resume-capable download
                    success = download_with_resume(url, filepath, chunk_size, timeout=60)
                else:
                    # Small file or no range support - regular download
                    response = requests.get(url, stream=True, timeout=30)
                    if response.status_code == 200:
                        total_size = int(response.headers.get('content-length', 0))
                        progress = tqdm(total=total_size, unit='iB', unit_scale=True, desc=filename)
                        with open(filepath, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=chunk_size):
                                if chunk:
                                    f.write(chunk)
                                    progress.update(len(chunk))
                        progress.close()
                        success = True
                    else:
                        success = False
                        print(f"Failed with status code: {response.status_code}")
                
                if success:
                    # Verify downloaded size
                    actual_size = os.path.getsize(filepath)
                    if remote_size > 0 and actual_size != remote_size:
                        print(f"WARNING: Downloaded size ({actual_size:,}) doesn't match expected ({remote_size:,})")
                        if attempt < max_retries - 1:
                            print("Retrying...")
                            continue
                    else:
                        print(f"Successfully downloaded {filename} ({actual_size:,} bytes)")
                        return
                    
            except (ChunkedEncodingError, ConnectionError, RequestException, Timeout) as e:
                print(f"Download failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    wait_time = min(60, 5 * (2 ** attempt))  # Exponential backoff, max 60s
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
    
    print(f"Failed to download {filename} after {max_retries} attempts")

def download_olmo_checkpoint(checkpoint_url, output_dir, wandb_project=None, wandb_entity=None):
    """
    Download OLMo checkpoint files from a given URL to the specified directory.
    
    Args:
        checkpoint_url (str): Base URL of the checkpoint
        output_dir (str): Directory to save the files
        wandb_project (str): Weights & Biases project name (optional)
        wandb_entity (str): Weights & Biases entity name (optional)
    """
    # wandb logging
    try:
        if wandb_entity is not None:
            import wandb
            os.environ["WANDB__SERVICE_WAIT"] = "6000"
            wandb.init(
                name="download_olmo_checkpoint.py",
                project=wandb_project,
                entity=wandb_entity,
                reinit='return_previous',  # chain this script with other scripts into a single wandb run
            )
    except ImportError:
        print("wandb not installed, skipping wandb logging.")
    
    # Files to download - now we check existence before attempting
    files = ['config.yaml', 'model.pt', 'optim.pt', 'train.pt', 'model.safetensors', 'optim.safetensors']
    
    base_url = checkpoint_url.rstrip('/')
    
    # Download each file
    for file in files:
        file_url = f"{base_url}/{file}"
        download_file(file_url, output_dir, file)
    
    print("\nDownload summary:")
    print(f"Output directory: {output_dir}")
    for file in files:
        filepath = os.path.join(output_dir, file)
        if os.path.exists(filepath):
            size_gb = os.path.getsize(filepath) / (1024**3)
            print(f"  ✓ {file}: {size_gb:.2f}GB")
        else:
            print(f"  ✗ {file}: Not downloaded")

def main():
    """Main function for command-line usage."""
    import argparse
    parser = argparse.ArgumentParser(description="Download OLMo checkpoint with robust handling for large files")
    parser.add_argument("checkpoint_url", type=str, help="Base URL of the checkpoint")
    parser.add_argument("output_dir", type=str, help="Directory to save the files")
    parser.add_argument("--wandb_project", type=str, default="download_olmo_checkpoint.py")
    parser.add_argument("--wandb_entity", type=str, default="sbordt-University of Tübingen")
    parser.add_argument("--no-chunked", action="store_true", help="Disable chunked downloads for large files")
    args = parser.parse_args()
    
    download_olmo_checkpoint(
        args.checkpoint_url, 
        args.output_dir, 
        args.wandb_project, 
        args.wandb_entity
    )

if __name__ == "__main__":
    main()
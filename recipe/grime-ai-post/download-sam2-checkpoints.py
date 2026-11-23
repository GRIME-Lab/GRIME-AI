#!/opt/anaconda1anaconda2anaconda3/bin/python

import hashlib
import io
import json
import os
import sys
import urllib.request
from pathlib import Path

# Download sources, tried in order. Override with environment variables if a host moves.
# Hugging Face hosts Meta's official SAM2.1 checkpoints (one repo per model size) and
# publishes each file's SHA256, which is used to verify downloads from either source.
HF_BASE_URL = os.environ.get("SAM2_HF_BASE_URL", "https://huggingface.co")
HF_ORG = os.environ.get("SAM2_HF_ORG", "facebook")
META_BASE_URL = os.environ.get("SAM2_META_BASE_URL", "https://dl.fbaipublicfiles.com/segment_anything_2/092824")


def hf_repo_for(filename):
    """sam2.1_hiera_base_plus.pt -> facebook/sam2.1-hiera-base-plus"""
    return f"{HF_ORG}/{Path(filename).stem.replace('_', '-')}"


def hf_expected_sha256(filename):
    """Return the SHA256 Hugging Face publishes for this file, or None if unavailable."""
    api_url = f"{HF_BASE_URL}/api/models/{hf_repo_for(filename)}/tree/main"
    try:
        with urllib.request.urlopen(api_url) as response:
            for entry in json.load(response):
                if entry.get("path") == filename and entry.get("lfs"):
                    return entry["lfs"].get("oid")
    except Exception as e:
        print(f"Could not get SHA256 for {filename} from Hugging Face: {e}")
    return None


def sha256_of(path):
    # Chunked read works on all supported Pythons (hashlib.file_digest needs 3.11+)
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(io.DEFAULT_BUFFER_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(filename, destination):
    """Try each source in order; keep the first download whose SHA256 matches."""
    expected_sha = hf_expected_sha256(filename)
    sources = [
        ("Hugging Face", f"{HF_BASE_URL}/{hf_repo_for(filename)}/resolve/main/{filename}"),
        ("Meta", f"{META_BASE_URL}/{filename}"),
    ]
    partial = f"{destination}.partial"
    errors = []

    for source_name, url in sources:
        print(f"Source: {source_name} ({url})")
        try:
            # Replicates 'curl -O' behavior
            urllib.request.urlretrieve(url, partial, reporthook=download_progress)
            print()
            if expected_sha:
                actual_sha = sha256_of(partial)
                if actual_sha != expected_sha:
                    raise ValueError(f"SHA256 mismatch (expected {expected_sha}, got {actual_sha})")
                print("SHA256 verified.")
            else:
                print("Warning: no published SHA256 available; file not verified.")
            os.replace(partial, destination)
            return
        except Exception as e:
            print(f"\n{source_name} failed: {e}")
            errors.append(f"{source_name}: {e}")
            if os.path.exists(partial):
                os.remove(partial)

    raise RuntimeError("; ".join(errors))

def download_progress(block_num, block_size, total_size):
    """Callback to display download progress."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 / total_size)
        # Using \r to overwrite the line for a clean progress output
        sys.stdout.write(f"\rDownloading... {percent:.1f}%")
        sys.stdout.flush()
    else:
        sys.stdout.write(f"\rDownloading... {downloaded} bytes")
        sys.stdout.flush()

def main():
    # 1. Locate the sam2 installation directory
    try:
        import sam2
        # Get the absolute path of the directory containing sam2
        sam2_init = Path(sam2.__file__).resolve()
        sam2_home = sam2_init.parent
    except ImportError:
        print("Error: 'sam2' package is not installed in the current environment.")
        sys.exit(1)

    # 2. Define the checkpoints directory
    checkpoint_home = Path(os.path.join(sam2_home,"checkpoints"))

    # 3. Create the directory if it doesn't exist
    try:
        checkpoint_home.mkdir(parents=True, exist_ok=True)
        print(f"Target directory: {checkpoint_home}")
    except Exception as e:
        print(f"Error: Could not create directory {checkpoint_home}. {e}")
        sys.exit(1)

    # 4. Define files (download sources are set at the top of this script)
    files = [
        "sam2.1_hiera_tiny.pt",
        "sam2.1_hiera_small.pt",
        "sam2.1_hiera_base_plus.pt",
        "sam2.1_hiera_large.pt"
    ]

    print("Starting download of SAM2.1 checkpoint files...")

    # 5. Download the files
    for filename in files:
        destination = os.path.join(checkpoint_home,filename)

        print(f"\nProcessing: {filename}")

        try:
            download_verified(filename, destination)
            print(f"Successfully downloaded to {destination}")
        except Exception as e:
            print(f"\nFailed to download {filename}: {e}")
            sys.exit(1)

    print("\n" + "="*40)
    print("Success! SAM2 checkpoint files are downloaded.")
    print("You may now run GRIME-AI.")
    print("="*40)

if __name__ == "__main__":
    main()

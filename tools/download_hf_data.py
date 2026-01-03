import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['HF_HUB_ETAG_TIMEOUT'] = '500'

from datasets import load_dataset
from tqdm import tqdm
import json
import time
import argparse
from huggingface_hub import hf_hub_download, snapshot_download


def download_repo(repo_id, cache_dir, local_dir):
    try:
        snapshot_download(repo_id=repo_id, 
                        local_dir=local_dir,
                        cache_dir=cache_dir,
                        local_dir_use_symlinks=False,
                        repo_type="dataset",
                        max_workers=16)
        return True
    except Exception as e:
        print(e)
        print("Sleep 300s and try again.")
        time.sleep(300)
        return False

def download(repo_id, cache_dir, local_dir):
    print(f"Downloading...")
    success = download_repo(repo_id, cache_dir, local_dir)
    idx = 0
    while not success and idx < 5:
        success = download_repo(repo_id, cache_dir, local_dir)
        idx += 1
    
    if success:
        print(f"done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a repository from Hugging Face Hub to a local directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Automatically show default values in help message
    )
    parser.add_argument(
        "--repo_id", 
        type=str, 
        default="lmms-lab/LLaVA-NeXT-Data",
        help="The ID of the Hugging Face repository to download."
    )
    parser.add_argument(
        "--cache_dir", 
        type=str, 
        default="~/.cache/cache_hf",
        help="The directory for Hugging Face to cache files."
    )
    parser.add_argument(
        "--local_dir", 
        type=str, 
        default="data/2D/",
        help="The root directory to save the downloaded files. Files will be stored under 'local_dir/repo_name'."
    )
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    _local_dir = os.path.join(args.local_dir, os.path.basename(args.repo_id))
    os.makedirs(_local_dir, exist_ok=True)
    download(args.repo_id, cache_dir=args.cache_dir, local_dir=_local_dir)

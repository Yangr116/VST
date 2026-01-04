import os
import argparse
from huggingface_hub import snapshot_download
from pathlib import Path

def download_model(model_name, cache_dir, local_dir_base):
    """
    Downloads a model from Hugging Face Hub to a specified local directory.
    """
    # Extract 'model_name' from 'organization/model_name' to use as a sub-directory name
    model_folder_name = model_name.split('/')[-1].strip()
    local_dir = os.path.join(local_dir_base, model_folder_name)

    if not os.path.exists(local_dir):
        print(f"Creating directory: {local_dir}")
        os.makedirs(local_dir)
    else:
        # Check if the directory is empty. If not, it might have been downloaded already.
        if os.listdir(local_dir):
            print(f"Directory {local_dir} already exists and is not empty. Skipping creation.")
        else:
            print(f"Directory {local_dir} already exists.")

    print(f"Starting download for {model_name}...")
    try:
        snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"Successfully downloaded {model_name} to {local_dir}")
    except Exception as e:
        print(f"An error occurred while downloading {model_name}: {e}")


if __name__ == "__main__":
    # 1. Initialize ArgumentParser
    parser = argparse.ArgumentParser(
        description="Download models from Hugging Face Hub.",
        formatter_class=argparse.RawTextHelpFormatter # Preserve help message formatting
    )

    # 2. Add arguments
    parser.add_argument(
        '--model_list',
        nargs='+',  # Indicates that one or more arguments can be accepted
        required=True,
        help="List of model names to download from Hugging Face Hub. \nExample: --model_list facebook/opt-125m google/gemma-7b"
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default=None, # If not specified, huggingface_hub will use the default cache directory
        help="Path to the cache directory for Hugging Face Hub downloads. \nIf not specified, the default HF cache will be used."
    )
    parser.add_argument(
        '--local_dir',
        type=str,
        required=True,
        help="The base local directory to save the downloaded models into. \nEach model will be saved in a subdirectory within this path."
    )

    # 3. Parse arguments
    args = parser.parse_args()

    # 4. Execute the main logic
    for model_name in args.model_list:
        download_model(model_name, args.cache_dir, args.local_dir)

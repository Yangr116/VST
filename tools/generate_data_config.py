import os
import yaml
import argparse

def find_parquet_directories(root_dir, target_ext=".parquet"):
    """
    Recursively traverses a directory to find all subdirectories
    containing files with the specified extension.
    """
    # Convert to absolute path to avoid errors with relative path calculations
    root_dir = os.path.abspath(root_dir)
    dir_info_list = []

    print(f"--- Start scanning: {root_dir} ---")

    # Key Change 1: followlinks=True allows os.walk to enter symbolic link directories
    for root, dirs, files in os.walk(root_dir, followlinks=True):

        # Debug print: Uncomment the line below to see which directories the script is traversing
        # print(f"Scanning: {root}")

        has_parquet = False
        for file in files:
            # Key Change 2: Case-insensitive extension matching
            if file.lower().endswith(target_ext.lower()):
                has_parquet = True
                # Once one is found, no need to continue checking files in this directory
                break

        if has_parquet:
            # Calculate the relative path from the root_dir
            rel_path = os.path.relpath(root, root_dir)

            # If it's the root directory itself, rel_path will be ".", change it to an empty string for cleaner path joining
            if rel_path == ".":
                rel_path = ""

            print(f"  [Target Found] Directory containing parquet: {rel_path if rel_path else '(root directory)'}")
            dir_info_list.append(dict(ann_path=rel_path, data_dir=root_dir))

    return dir_info_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find directories containing Parquet files and generate a YAML file.")
    parser.add_argument("data_dir", type=str, help="The root directory path for the data.")
    parser.add_argument("save_path", type=str, help="The root directory path for the data.")
    args = parser.parse_args()

    data_dir = args.data_dir

    if not os.path.exists(data_dir):
        print(f"Error: Path does not exist: {data_dir}")
        exit(1)

    yaml_info = find_parquet_directories(data_dir)

    print(f"--- Scan finished. Found {len(yaml_info)} directories in total. ---")

    if len(yaml_info) > 0:
        # Saving logic
        save_path = args.save_path

        print(f"Saving info to: {save_path}")
        with open(save_path, 'w') as f:
            # Use default_flow_style=False for a more readable block style in YAML
            yaml.dump(yaml_info, f, default_flow_style=False)
    else:
        print("No directories containing .parquet files were found. Please check the path or file extension.")

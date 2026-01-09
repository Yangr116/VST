import argparse
from pyarrow import parquet as pq
from glob import glob
import json
import os
from PIL import Image
import io
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Global variables for workers to access
# These will be set in the main block before workers start
OUTPUT_DIR = ""
IMAGE_OUTPUT_DIR = ""

def setup_directories(out_dir):
    """Create necessary output directories if they don't exist."""
    img_dir = os.path.join(out_dir, "images")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    return img_dir

def process_single_parquet(parquet_file):
    """
    Process a single parquet file: extract images and format JSON data.
    Returns a list of processed JSON objects.
    """
    local_results = []
    
    # Note: Workers inherit global variables from the parent process in fork-based systems (Linux/Mac).
    # For spawn-based systems (Windows), arguments might need to be passed explicitly, 
    # but globals usually work for read-only config in simple scripts.
    
    try:
        parquet_data = pq.ParquetFile(parquet_file)
        
        # Iterate through batches to manage memory usage
        for batch in parquet_data.iter_batches(batch_size=100):
            batch_data = batch.to_pylist()
            
            for row in batch_data:
                # Extract and handle images
                images = row.pop('images')
                json_sample = row.copy()
                image_info_list = []
                
                # If type is text, ignore images
                if row.get('type') == 'text':
                    images = []
                
                for idx, img_obj in enumerate(images):
                    # Handle potential typo in keys ('bytes' vs 'byres')
                    img_bytes = img_obj.get('bytes') or img_obj.get('byres')
                    
                    if img_bytes:
                        img_filename = f"{row['id']}_{idx}.jpg"
                        # Use the global IMAGE_OUTPUT_DIR
                        img_save_path = os.path.join(IMAGE_OUTPUT_DIR, img_filename)
                        # Store relative path for JSON
                        image_info_list.append(os.path.join("images", img_filename))
                        
                        try:
                            # Convert bytes to image and save
                            image = Image.open(io.BytesIO(img_bytes))
                            image = image.convert("RGB")
                            image.save(img_save_path)
                        except Exception as e:
                            print(f"Error saving image {img_filename}: {e}")
                
                json_sample['images'] = image_info_list

                # Handle image placeholders in conversations
                convs = json_sample['conversations']
                IMAGE_FLAG = "<|image_pad|>"
                image_count = len(json_sample['images'])
                
                # Check if placeholders match image count in human prompts
                human_value = "".join([conv['value'] for conv in convs if conv['from'] == 'human'])
                
                if human_value.count(IMAGE_FLAG) != image_count:
                    # Ensure the first message is from human before prepending
                    if convs and convs[0]['from'] == 'human':
                        # Remove existing flags and prepend the correct number of flags
                        clean_value = convs[0]['value'].replace(IMAGE_FLAG, '')
                        convs[0]['value'] = (IMAGE_FLAG * image_count) + clean_value

                # NOTE: you can replace our image flag to your image flag
                # convs[0]['value'] = convs[0]['value'].replace(IMAGE_FLAG, 'your image flag')
                
                local_results.append(json_sample)
                
    except Exception as e:
        print(f"Error processing file {parquet_file}: {e}")
        
    return local_results

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process Parquet files to extract images and JSON.")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        required=True, 
        help="Path to the directory containing .parquet files (e.g., /path/to/vst_500k)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data", 
        help="Path to the output directory (default: 'data')"
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=max(1, cpu_count() - 2), 
        help="Number of worker processes to use"
    )
    return parser.parse_args()

def main():
    global OUTPUT_DIR, IMAGE_OUTPUT_DIR
    
    args = parse_arguments()
    
    # Set paths from arguments
    data_dir = args.data_dir
    OUTPUT_DIR = args.output_dir
    
    # Setup directories and set global IMAGE_OUTPUT_DIR
    IMAGE_OUTPUT_DIR = setup_directories(OUTPUT_DIR)
    
    final_json_path = os.path.join(OUTPUT_DIR, "vst_500k.json")
    
    # Find all parquet files
    parquet_files = glob(os.path.join(data_dir, "**/*.parquet"), recursive=True)
    
    if not parquet_files:
        print(f"No parquet files found in {data_dir}")
        return

    print(f"Found {len(parquet_files)} parquet files in '{data_dir}'.")
    print(f"Starting processing with {args.num_workers} workers...")
    print(f"Output will be saved to '{OUTPUT_DIR}'")

    all_results = []
    
    # Use multiprocessing Pool
    with Pool(processes=args.num_workers) as pool:
        # imap_unordered yields results as soon as they are ready
        for result_batch in tqdm(pool.imap_unordered(process_single_parquet, parquet_files), total=len(parquet_files)):
            all_results.extend(result_batch)

    print(f"Processing complete. Saving {len(all_results)} records to JSON...")
    
    # Save the final consolidated JSON
    with open(final_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4, default=str)
        
    print(f"Saved to {final_json_path}")

if __name__ == "__main__":
    main()

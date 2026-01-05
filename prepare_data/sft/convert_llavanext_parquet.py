import copy
import argparse
import os
import io
import time
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import datasets as hf_datasets
from loguru import logger
from qwen_vl_utils.vision_process import fetch_image, to_rgb, smart_resize
import re
import random
import uuid
from concurrent.futures import ThreadPoolExecutor
import yaml
import jsonlines
from glob import glob
import gc
import sys


def build_convs(convs):
    new_convs = []
    for conv in convs:
        new_convs.append({'from': conv['from'], 'value': conv['value']})
    return new_convs


def encode_image_to_bytes(img):
    with io.BytesIO() as buffer:
        img.save(buffer, format='PNG')
        return buffer.getvalue()


def resize_image(image: Image.Image):
    orig_width, orig_height = image.size

    new_width, new_height = orig_width, orig_height
    if orig_width < 28 or orig_height < 28:
        scale = max(56.0 / orig_width, 56.0 / orig_height)
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
    
    if 100.0 < new_width / new_height or new_width / new_height < 1 / 100.0:
        logger.error(f"extream scale ratio {new_width / new_height}")
        return None

    if (new_width, new_height) != (orig_width, orig_height):
        resized_img = image.resize((new_width, new_height))
    else:
        resized_img = image

    meta_info = {
        'resized_width': new_width if new_width != orig_width else -1,
        'resized_height': new_height if new_height != orig_height else -1,
        'width': orig_width,
        'height': orig_height
    }
    return resized_img, meta_info


def process_item(item_with_info):
    """
    Process a single data item by converting its image to binary format.
    
    Args:
        item_with_info: Tuple containing (item, image_dir, data_source)
    
    Returns:
        Processed item or None if processing fails
    """
    item, image_dir, data_source = item_with_info
    
    try:
        _image_path = item.get('image', item.get('images'))
        if _image_path is None:
            return {
                'conversations': build_convs(item['conversations']),
                'id': str(item['id']),
                'data_source': item.get('data_source', data_source),
                'images': [{'bytes': b'', 'path': ''}],
                'type': "text",
                'meta_info': json.dumps([{'resized_width': -1, 'resized_height': -1, 'width': -1, 'height': -1}])
            }
        if not isinstance(_image_path, list):
            _image_path = [_image_path]
        data_type = item.get('type', 'unknow')
        item_id = str(item.get('id', uuid.uuid4()))
        
        images = []
        meta_info_list = []
        for _image_path_item in _image_path:
            if not isinstance(_image_path_item, Image.Image):
                # load image
                pass
            else:
                image = _image_path_item
                path = f"{item_id}.jpg"
            resized_img, meta_info = resize_image(image)
            image_binary = encode_image_to_bytes(resized_img)
            images.append({'bytes': image_binary, 'path': path})
            meta_info_list.append(meta_info)

        # Create a new item with only the required fields
        return {
            'conversations': build_convs(item['conversations']),
            'id': item_id,
            'data_source': item.get('data_source', data_source),
            'images': images,  # list
            'type': data_type,
            'meta_info': json.dumps(meta_info_list), # list
        }
    except Exception as e:
        logger.error(f"Error processing item {item['conversations']}\nerror: {e}")
        return None


def convert_parquet_to_parquet(args):
    """
    Convert a batch of JSON data to Parquet format
    
    Args:
        batch_info: Tuple containing (data_batch, output_path, image_dir, batch_idx)
        
    Returns:
        Tuple of (batch_idx, output_path, number of processed items)
    """
    batch_info, save_batch_size = args
    data_batch, output_path, image_dir, batch_idx = batch_info

    if not isinstance(data_batch, list):
        data_batch = [data_batch]

    data_batch = hf_datasets.load_dataset("parquet", data_files=data_batch, split='train', streaming=True)

    processed_data = []
    
    for i, item in enumerate(tqdm(data_batch, desc=f"Batch-{batch_idx}", position=batch_idx)):
        result = process_item((item, image_dir, output_path))
        if result is not None:
            processed_data.append(result)

    # save
    if processed_data:
        logger.info(f"Batch {batch_idx}: Saving {len(processed_data)} items to {output_path}")
        dataset = hf_datasets.Dataset.from_list(processed_data)
        dataset.to_parquet(output_path, batch_size=save_batch_size)

        del dataset
        gc.collect()
        
        return batch_idx, output_path, len(processed_data)
    else:
        logger.warning(f"Batch {batch_idx}: No valid items to save to {output_path}")
        return batch_idx, output_path, 0
        

def convert_parquet_to_parquet_try(args):
    try:
        return convert_parquet_to_parquet(args)
    except Exception as e:
        data_info = args[0][0]
        if isinstance(data_info, list):
            data_info = data_info[0]
        logger.error(f"Error processing batch {args[0][-1]}, {data_info}: {str(e)}")
        gc.collect()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Prepare Parquet")

    parser.add_argument('--data_dir', '-d', required=True)

    parser.add_argument("--output_dir", "-o", required=True,
                        help="Directory to save the output Parquet files")

    parser.add_argument("--tag", type=str, required=True,
                        help="the tag of this data")

    parser.add_argument("--image_dir", "-i", default="",
                        help="Directory containing the images referenced in the JSON data")

    parser.add_argument("--workers", "-w", type=int, default=64,
                        help="Number of worker processes for batch processing (default: auto)")

    parser.add_argument("--save_batch_size", type=int, default=100,
                        help="Number of items to save into parquet")
    
    parser.add_argument("--debug", action='store_true', default=False, help="enable debug")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    data_dir = args.data_dir
    output_dir = args.output_dir
    num_workers = args.workers
    SAVE_BATCH_SIZE = args.save_batch_size
    image_dir = args.image_dir
    tag = args.tag
    output_dir = os.path.join(output_dir, tag)
    os.makedirs(output_dir, exist_ok=True)

    parquet_list = glob(os.path.join(data_dir, "*.parquet"))
    parquet_list = parquet_list[:10]
    
    debug = args.debug
    if debug:
        debug_args = (("LLaVA-NeXT-Data/data/train-00008-of-00250.parquet", os.path.join(output_dir, f"{tag}_0.parquet"), "", 0), 100)
        convert_parquet_to_parquet(debug_args)
        sys.exit()

    # prepare batch info
    batch_infos = []
    for idx, batch_data in enumerate(parquet_list):
        output_path = os.path.join(output_dir, f"{tag}_{idx}.parquet")
        batch_infos.append((batch_data, output_path, image_dir, idx))

    results = []
    mp_context = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=mp_context) as executor:
        futures = [executor.submit(convert_parquet_to_parquet_try, (info, SAVE_BATCH_SIZE)) for info in batch_infos]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
            try:
                results.append(future.result())
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")

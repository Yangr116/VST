"""
For video data, there are two means
1. with original video files, such as *.mp4, directly convert them into bytes, `type` is video, `path` is *.mp4, *.avi and so on
2. consists of many images, such as sharegpt4video, we save them into list, `type` is video, `path` is *.png, ...
"""

import ujson as json
import copy

import argparse
import os
import io
import time
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
import yaml
import jsonlines
import psutil
import gc
import resource


def load_json_data(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    return data


def build_convs(convs):
    new_convs = []
    for conv in convs:
        new_convs.append({'from': conv['from'], 'value': conv['value']})
    return new_convs


def read_file_safely(file_path, max_size_mb=500):
    """
    安全读取文件，避免内存溢出
    """
    try:
        file_size = os.path.getsize(file_path)
        if file_size > max_size_mb * 1024 * 1024:
            logger.warning(f"File too large ({file_size / 1024 / 1024:.1f}MB): {file_path}")
        
        with open(file_path, 'rb') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return None


def process_item(item_with_info):
    item, video_dir, data_source = item_with_info

    _vision_path = item.get('video', None)
    assert _vision_path is not None
    if not isinstance(_vision_path, list):
        _vision_path = [_vision_path]
    
    vision_path = []
    for _vision_path_item in _vision_path:
        if video_dir != "":
            vision_path_item = os.path.join(video_dir, _vision_path_item.strip('/'))
        else:
            vision_path_item = os.path.join(video_dir, _vision_path_item)
        if not os.path.exists(vision_path_item):
            logger.warning(f"vision file is not found: {vision_path_item}")
            return None
        else:
            vision_path.append(vision_path_item)
    
    data_type = item['type']
    image_info_list = []
    meta_info_list = []
    
    for file_path in vision_path:
        # 使用安全读取函数
        video_bytes = read_file_safely(file_path)
        if video_bytes is None:
            logger.warning(f"Failed to read video file: {file_path}")
            return None
            
        image_info_list.append({'bytes': video_bytes, 'path': file_path})
        # for video, we doesn't record the information
        meta_info = {
            'resized_width': -1,
            'resized_height': -1,
            'width': -1,
            'height': -1
        }
        meta_info_list.append(meta_info)
    
    builded_item = {
        "conversations": build_convs(item['conversations']),
        'id': str(item.get('id', uuid.uuid4())),
        'data_source': item.get('data_source', data_source),
        'images': image_info_list,
        'type': data_type,
        'meta_info': json.dumps(meta_info_list),
    }
    return builded_item


def process_item_with_try(item_with_info):
    try:
        return process_item(item_with_info)
    except Exception as e:
        logger.error(f"Error processing item: {e}")
        return None


def monitor_memory():
    """监控内存使用情况"""
    memory = psutil.virtual_memory()
    return memory.percent, memory.available / (1024**3)  # GB


def convert_batch_to_parquet(args):
    """
    Convert a batch of JSON data to Parquet format
    
    Args:
        batch_info: Tuple containing (data_batch, output_path, video_dir, batch_idx)
        
    Returns:
        Tuple of (batch_idx, output_path, number of processed items)
    """
    batch_info, save_batch_size = args
    data_batch, output_path, video_dir, batch_idx = batch_info
    
    # 监控内存使用
    memory_usage, available_gb = monitor_memory()
    logger.info(f"Batch {batch_idx}: Starting processing {len(data_batch)} items to {output_path}")
    logger.info(f"Memory usage: {memory_usage:.1f}%, Available: {available_gb:.1f}GB")
    
    try:
        if isinstance(data_batch, dict):
            data_batch = [data_batch]
        
        # 移除线程池嵌套，直接串行处理以减少内存压力
        processed_data = []
        
        for i, item in enumerate(tqdm(data_batch, desc=f"Batch-{batch_idx}", position=batch_idx)):
            # 定期检查内存使用
            if i % 50 == 0:
                memory_usage, available_gb = monitor_memory()
                if memory_usage > 85:  # 内存使用超过85%时警告
                    logger.warning(f"High memory usage: {memory_usage:.1f}%")
                    gc.collect()  # 强制垃圾回收
            
            result = process_item_with_try((item, video_dir, output_path))
            if result is not None:
                processed_data.append(result)

        # 保存数据
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
            
    except Exception as e:
        logger.error(f"Error processing batch {batch_idx}: {str(e)}")
        gc.collect()  # 发生异常时也要清理内存
        return batch_idx, output_path, 0


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Convert JSON data with images to Parquet format")

    parser.add_argument('-j', '--json_file', type=str, nargs='+', required=True, help='One or more JSON files')

    parser.add_argument("--output_dir", "-o", required=True,
                        help="Directory to save the output Parquet files")
    
    parser.add_argument("--video_dir", "-i", default="",
                        help="Directory containing the images referenced in the JSON data")
    
    parser.add_argument("--batch_size", "-b", type=int, default=5000,
                        help="Number of items to process in each batch (default: 5000)")

    parser.add_argument("--max_batches", "-m", type=int, default=None,
                        help="Maximum number of batches to process (default: process all)")
    
    parser.add_argument("--workers", "-w", type=int, default=64,
                        help="Number of worker processes for batch processing (default: auto)")

    parser.add_argument("--save_batch_size", type=int, default=100,
                        help="Number of items to save into parquet")

    parser.add_argument("--tag", type=str, required=True,
                        help="the tag of this data")

    parser.add_argument("--debug", action='store_true', default=False, help="enable debug")
    return parser.parse_args()


def main():
    args = parse_arguments()
    jsonfiles = args.json_file
    output_dir = args.output_dir
    video_dir = args.video_dir
    tag = args.tag
    batch_size = args.batch_size
    num_workers = args.workers
    output_dir = os.path.join(output_dir, tag)
    os.makedirs(output_dir, exist_ok=True)

    global SAVE_BATCH_SIZE
    SAVE_BATCH_SIZE = args.save_batch_size

    print("Loading data...")
    data = []
    video_flag = "<|video_pad|>"  # NOTE: Video data MUST set this flag
    for jsonfile in jsonfiles:
        ori_data = load_json_data(jsonfile)
        
        for idx, sample in enumerate(ori_data):
            convs = sample['conversations']
            video_file = sample.get('video', None)
            assert video_file is not None
            for conv_id, conv in enumerate(convs):
                if "<image>" in conv['value'] and conv['from'] == 'human':
                    conv["value"] = conv["value"].replace('<image>\n', '<image>').replace('\n<image>', '<image>').replace('<image>', video_flag)
                elif "<image>" not in conv['value'] and conv['from'] == 'human' and conv_id == 0 and video_file is not None:
                    conv["value"] = video_flag + conv["value"]
            
            if sample.get('type') != 'video':
                sample['type'] = 'video'

            if sample.get('data_source') is None:
                sample['data_source'] = jsonfile
            data.append(sample)
    
    print(f"Loaded {len(data)} samples")
    # import pdb; pdb.set_trace()
    random.shuffle(data)
    
    # 调整参数以减少内存压力
    max_batches = None
    start_time = time.time()
    
    debug = args.debug
    if debug:
        processed_data = []
        output_path = os.path.join(output_dir, f"try_0.parquet")
        for sample in tqdm(data[:100]):  # 减少调试数据量
            data_item = process_item((sample, video_dir, sample['data_source']))
            if data_item is not None:
                processed_data.append(data_item)
        if processed_data:
            dataset = hf_datasets.Dataset.from_list(processed_data)
            dataset.to_parquet(output_path)
        return

    num_samples = len(data)
    num_batches = (num_samples + batch_size - 1) // batch_size
    if max_batches is not None:
        num_batches = min(num_batches, max_batches)
    logger.info(f"Processing {num_samples} samples in {num_batches} batches (batch size: {batch_size})")

    # Check if the last batch is small
    last_batch_size = num_samples % batch_size
    if last_batch_size > 0 and last_batch_size < batch_size // 2 and num_batches > 1:
        # 如果最后一批太小，合并到前一批
        effective_num_batches = num_batches - 1
        logger.info(f"Last batch size ({last_batch_size}) too small, merging with previous batch")
    else:
        effective_num_batches = num_batches

    # Prepare batch info
    batch_infos = []
    for idx in range(effective_num_batches):
        start_idx = idx * batch_size

        # If this is the last batch and we're combining with the small last batch
        if idx == effective_num_batches - 1 and effective_num_batches < num_batches:
            end_idx = num_samples
        else:
            end_idx = min(start_idx + batch_size, num_samples)

        output_path = os.path.join(output_dir, f"{tag}_{idx}.parquet")
        batch_data = data[start_idx:end_idx]
        batch_infos.append((batch_data, output_path, video_dir, idx))
    
    # 显示系统信息
    memory_usage, available_gb = monitor_memory()
    logger.info(f"System memory: {memory_usage:.1f}% used, {available_gb:.1f}GB available")
    logger.info(f"Using {num_workers} worker(s) for processing")
    
    # Process batches using multiprocessing
    results = []
    if num_workers == 1:
        # 串行处理，避免多进程内存问题
        logger.info("Using serial processing to avoid memory issues")
        for info in tqdm(batch_infos, desc="Processing batches"):
            result = convert_batch_to_parquet((info, SAVE_BATCH_SIZE))
            results.append(result)
    else:
        # 多进程处理
        mp_context = mp.get_context('spawn')
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=mp_context) as executor:
            futures = [executor.submit(convert_batch_to_parquet, (info, SAVE_BATCH_SIZE)) for info in batch_infos]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")
    
    elapsed_time = time.time() - start_time
    successful_batches = len([r for r in results if r[2] > 0])
    total_items = sum(r[2] for r in results)
    
    logger.info(f"Processing complete!")
    logger.info(f"Total time: {elapsed_time:.2f} seconds")
    logger.info(f"Successful batches: {successful_batches}/{len(batch_infos)}")
    logger.info(f"Total items processed: {total_items}")


if __name__ == "__main__":
    """
    python prepare_llavavideo2parquet.py --json_file video_ann.json \
    --output_dir "data/parquet/video" \
    --video_dir "video_save_dir" \
    --workers 16 \
    --batch_size 1000 \
    --save_batch_size 10 \
    --tag "debug"
    """
    main()


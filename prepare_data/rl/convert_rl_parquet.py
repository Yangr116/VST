"""
[
'images' (list), 
'problem' (str), 
'answer' (str)
]
"""
import argparse
import ujson as json
import os
import uuid
from tqdm import tqdm
import random
import multiprocessing as mp
from functools import partial
from PIL import Image

from loguru import logger
from vst.preprocess import IMAGE_FLAG
from qwen_vl_utils.vision_process import to_rgb, smart_resize

from prepare_data.utils import (encode_image_to_bytes, save_to_parquet, split_into_chunks, 
                                save_to_parquet, random_resize_image_pil_list, check_image_size)



MIN_PIXELS = 4*28*28
MAX_PIXELS = 4096 * 28 * 28 # single image pixels


def process_single_sample(sample, image_dir, jsonfile, resize=False):
    image_path_list = sample['images']
    if not isinstance(image_path_list, list):
        image_path_list = [image_path_list]
    image_path_list = [os.path.join(image_dir, x) for x in image_path_list]
    image_pil_list = [Image.open(x).convert('RGB') for x in image_path_list]
    # resize
    if resize:
        image_pil_list = random_resize_image_pil_list(image_pil_list)
    
    # keep image size
    _image_pil_list = []
    for image_pil in image_pil_list:
        width, height = image_pil.size
        resized_height, resized_width = smart_resize(
            height,
            width,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS,
        )
        image_pil = image_pil.resize((resized_width, resized_height))
        _image_pil_list.append(image_pil)
    
    image_pil_list = [check_image_size(x) for x in _image_pil_list]
    image_tokens = [x.size[0] * x.size[1] / (28*28) for x in image_pil_list]
    image_tokens = sum(image_tokens)
    assert image_tokens < 8192, f"image is too long {image_tokens}"
    
    images = [{'bytes': encode_image_to_bytes(image_pil), 'path': image_path} for image_pil, image_path in zip(image_pil_list, image_path_list)]

    data_source = jsonfile

    question = sample['question']
    if IMAGE_FLAG not in question:
        question = IMAGE_FLAG * len(image_path_list) + question
    assert question.count(IMAGE_FLAG) == len(image_path_list)

    answer = str(sample['answer']).strip()

    uid = sample.get('uuid', str(uuid.uuid4()))
    meta_info = sample.get("meta_info", None)
    if meta_info is None:
        meta_info = {
            "data_source": sample.get('data_source', data_source)
        }
    if not isinstance(meta_info, str):
        meta_info = json.dumps(meta_info)
    parquet_item = dict(
        id=uid,
        images=images,
        question=question,
        answer=answer,
        meta_info=meta_info
    )
    return [], parquet_item


def process_single_sample_with_try(*args, **kwargs):
    try:
        return process_single_sample(*args, **kwargs)
    except Exception as e:
        print(f"Error processing sample: {e}")
        return [], None


def process_chunk(chunk_data, chunk_idx, image_dir, jsonfile, parquet_dir, ann_save_name, resize=False):
    json_results = []
    parquet_results = []
    
    print(f"Processing chunk {chunk_idx} with {len(chunk_data)} samples...")
    
    for sample in tqdm(chunk_data, desc=f"Chunk {chunk_idx}"):
        json_result, parquet_result = process_single_sample_with_try(
            sample, 
            image_dir, 
            jsonfile,
            resize=resize
            )
        
        if parquet_result is not None:
            if isinstance(parquet_result, list):
                parquet_results.extend(parquet_result)
            else:
                parquet_results.append(parquet_result)
    
    if parquet_results:
        parquet_path = os.path.join(parquet_dir, f'{ann_save_name}_{chunk_idx}.parquet')
        save_to_parquet(parquet_results, parquet_path)
    return json_results


if __name__ == "__main__":
    """
    python prepare_data/rl/convert_rl_parquet.py \
        -j /opt/tiger/rayyang/spatial/20250610/cam-obj-relation/scannet-reason_seedvlm.json \
        -i /opt/tiger/rayyang/spatial/20250610/cam-obj-relation/images \
        --parquet /opt/tiger/rayyang/spatial/data/inhouse/parquet/multi_image/reason \
        -t multi_view_reason_ziyu_20250610_cam-obj-relation
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--jsonfile', type=str, nargs='+', required=True)
    parser.add_argument('-i', '--image_dir', type=str, default="dataset/image_dir")
    parser.add_argument('-pd', '--parquet_dir', type=str, default="dataset/parquet/rl")
    parser.add_argument('-t', '--tag', type=str, required=True)
    parser.add_argument('--chunk_size', type=int, default=500)
    parser.add_argument('--resize', action='store_true')
    args = parser.parse_args()
    jsonfiles = args.jsonfile
    image_dir = args.image_dir
    parquet_dir = args.parquet_dir
    tag = args.tag
    ann_save_name = tag.replace('/', '_')
    parquet_dir = os.path.join(parquet_dir, ann_save_name)
    os.makedirs(parquet_dir, exist_ok=True)
    chunk_size = args.chunk_size
    
    data = []
    for jsonfile in jsonfiles:
        data_item = json.load(open(jsonfile))
        data.extend(data_item)
    random.shuffle(data)
    print(f"Loaded {len(data)} samples")
    chunks = split_into_chunks(data, chunk_size=chunk_size)

    num_processes = min(mp.cpu_count(), len(chunks))
    print(f"Using {num_processes} processes")

    with mp.Pool(processes=num_processes) as pool:
        process_func = partial(
            process_chunk,
            image_dir=image_dir,
            jsonfile=jsonfile,
            parquet_dir=parquet_dir,
            ann_save_name=ann_save_name,
            resize=args.resize
        )
        
        args_list = [(chunk, idx) for idx, chunk in enumerate(chunks)]
        results = pool.starmap(process_func, args_list)

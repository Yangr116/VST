import io
import os
import sys
import uuid
from tqdm import tqdm
import random
import multiprocessing as mp
from functools import partial
from PIL import Image
from loguru import logger
import gc
import ujson as json
import datasets as hf_datasets
import jsonlines
from qwen_vl_utils.vision_process import fetch_image, to_rgb, smart_resize
from prepare_data.utils import encode_image_to_bytes, resize_image, split_into_chunks


def build_convs(convs):
    new_convs = []
    for conv in convs:
        new_convs.append({'from': conv['from'], 'value': conv['value']})
    return new_convs


def load_jsonl_data(json_file_path):
    with jsonlines.open(json_file_path, 'r') as objs:
        data = []
        for obj in objs:
            data.append(obj)
    return data


def process_single_sample(sample, image_dir, jsonfile, data_type='mm_qa'):
    image_path_list = sample.get("images", sample.get("image"))
    assert image_path_list is not None
    if not isinstance(image_path_list, list):
        image_path_list = [image_path_list]
    image_path_list = [os.path.join(image_dir, x) for x in image_path_list]
    image_pil_list = []
    meta_info_list = []
    for image_path in image_path_list:
        image_pil = Image.open(image_path).convert('RGB')
        resized_img, meta_info = resize_image(image_pil)
        image_pil_list.append(resized_img)
        meta_info_list.append(meta_info)
    # resize
    images = [{'bytes': encode_image_to_bytes(image_pil), 'path': image_path} for image_pil, image_path in zip(image_pil_list, image_path_list)]
    conversations = build_convs(sample['conversations']),

    uid = sample.get('uuid', str(uuid.uuid4()))
    parquet_item = dict(
        id=uid,
        images=images,
        conversations=conversations,
        type=data_type,
        data_source=sample.get('data_source', jsonfile),
        meta_info=json.dumps(meta_info_list),  # meta info should be a list
    )
    return [], parquet_item


def process_single_sample_with_try(*args, **kwargs):
    try:
        return process_single_sample(*args, **kwargs)
    except Exception as e:
        print(f"Error processing sample: {e}")
        return [], None


def process_chunk(chunk_data, chunk_idx, image_dir, jsonfile, output_dir, ann_save_name, save_batch_size=100):

    json_results = []
    parquet_results = []
    
    print(f"Processing chunk {chunk_idx} with {len(chunk_data)} samples...")
    
    for sample in tqdm(chunk_data, desc=f"Chunk {chunk_idx}"):
        json_result, parquet_result = process_single_sample_with_try(
            sample,
            image_dir,
            jsonfile,
            )
        
        if parquet_result is not None:
            if isinstance(parquet_result, list):
                parquet_results.extend(parquet_result)
            else:
                parquet_results.append(parquet_result)
    
    # 保存parquet文件
    if parquet_results:
        parquet_path = os.path.join(output_dir, f'{ann_save_name}_{chunk_idx}.parquet')
        logger.info(f"Batch {chunk_idx}: Saving {len(chunk_data)} items to {parquet_path}")
        dataset = hf_datasets.Dataset.from_list(parquet_results)
        dataset.to_parquet(parquet_path, batch_size=save_batch_size)

        del dataset
        gc.collect()
        
    return json_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-j', '--jsonfile', type=str, nargs='+', required=True)
    
    parser.add_argument('-i', '--image_dir', type=str, default="")
    
    parser.add_argument("--output_dir", "-o", required=True,
                        help="Directory to save the output Parquet files")
    
    parser.add_argument("--tag", type=str, required=True,
                        help="the tag of this data")
    
    parser.add_argument('--chunk_size', type=int, default=500)

    parser.add_argument("--workers", "-w", type=int, default=64,
                        help="Number of worker processes for batch processing (default: auto)")

    parser.add_argument("--save_batch_size", type=int, default=100,
                        help="Number of items to save into parquet")
    
    parser.add_argument("--debug", action='store_true', default=False, help="enable debug")
    
    args = parser.parse_args()
    jsonfiles = args.jsonfile
    image_dir = args.image_dir
    output_dir = args.output_dir
    tag = args.tag
    ann_save_name = tag.replace('/', '_')
    output_dir = os.path.join(output_dir, ann_save_name)
    os.makedirs(output_dir, exist_ok=True)
    chunk_size = args.chunk_size

    data = []
    for jsonfile in jsonfiles:
        if jsonfile.endswith('.jsonl'):
            data_item = load_jsonl_data(jsonfile)
        else:
            data_item = json.load(open(jsonfile))
        data.extend(data_item)
    random.shuffle(data)
    print(f"Loaded {len(data)} samples")
    chunks = split_into_chunks(data, chunk_size=chunk_size)

    num_processes = min(mp.cpu_count(), len(chunks))
    print(f"Using {num_processes} processes")

    if args.debug:
        process_single_sample(chunks[0][0], image_dir, jsonfile)
        sys.exit()

    with mp.Pool(processes=num_processes) as pool:

        process_func = partial(
            process_chunk,
            image_dir=image_dir,
            jsonfile=jsonfile,
            output_dir=output_dir,
            ann_save_name=ann_save_name,
            save_batch_size=args.save_batch_size
        )
        
        # 处理所有chunks
        args_list = [(chunk, idx) for idx, chunk in enumerate(chunks)]
        results = pool.starmap(process_func, args_list)

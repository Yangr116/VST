import os
import random
import json
from glob import glob
from pathlib import Path
from PIL import Image
import h5py
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np
import datasets as hf_datasets
import argparse
import io


def encode_image_to_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


def _normalize_quantile(x):
    q01, q99 = ACTION_Q01, ACTION_Q99
    return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def process_file(datafile):
    results = []
    try:
        with h5py.File(datafile, "r") as f:
            data = f['data']
            for i in range(len(data.keys())):
                demo_key = f"demo_{i}"
                if demo_key not in data:
                    continue
                demo_data = data[demo_key]
                actions = demo_data['actions']
                obs = demo_data['obs']
                agentview_rgb = obs['agentview_rgb']
                for idx, (action, agentview_rgb_item) in enumerate(zip(actions, agentview_rgb)):
                    # build item
                    img = Image.fromarray(agentview_rgb_item[::-1,::-1])  # rotate
                    uid = [str(Path(datafile).parent.stem), str(Path(datafile).stem), str(demo_key), str(idx)]
                    uid = "_".join(uid)
                    images = [{'bytes': encode_image_to_bytes(img), 'path': f"{uid}.png"}]
                    words =  str(Path(datafile).name)
                    words = words[:-10].split('_')
                    command = ''
                    for w in words:
                        if "SCENE" in w:
                            command = ''
                            continue
                        command = command + w + ' '
                    command = command[:-1]
                    action_norm = _normalize_quantile(action)
                    # import pdb; pdb.set_trace()
                    meta_info = dict(
                        action=action_norm.tolist(), 
                        action_unnorm=action.tolist(), 
                        language_instruction=command, 
                        data_source=datafile)
                    meta_info = json.dumps(meta_info)
                    parquet_item = dict(
                        id=uid,
                        images=images,
                        meta_info=meta_info,
                    )
                    results.append(parquet_item)
    except Exception as e:
        print(f"Error processing {datafile}: {e}")
    return results


def save_to_parquet(data, parquet_path):
    dataset = hf_datasets.Dataset.from_list(data)
    dataset.to_parquet(parquet_path)
    print(f"Saved parquet with {len(data)} samples to {parquet_path}")


def chunk_list_gen(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--libero_dir', type=str, required=True)
    args = parser.parse_args()

    _save_dir = args.save_dir
    LIBERO_DIR = args.libero_dir

    data_dir_list = [
        (f"{LIBERO_DIR}/libero_goal_no_noops", "prepare_data/vla/libero/norm_stats_libero_goal.json"),
        (f"{LIBERO_DIR}/libero_object_no_noops", "prepare_data/vla/libero/norm_stats_libero_object.json"),
        (f"{LIBERO_DIR}/libero_spatial_no_noops", "prepare_data/vla/libero/norm_stats_libero_spatial.json"),
        (f"{LIBERO_DIR}/libero_10_no_noops", "prepare_data/vla/libero/norm_stats_libero_10.json")
    ]

    for (data_dir, norm_state_file) in data_dir_list:

        NORM_STATES = json.load(open(norm_state_file))['norm_stats']
        ACTION_MEAN = np.array(NORM_STATES['action']['mean'])
        ACTION_STD = np.array(NORM_STATES['action']['std'])
        ACTION_Q01 = np.array(NORM_STATES['action']['q01'])
        ACTION_Q99 = np.array(NORM_STATES['action']['q99'])

        datafiles = glob(os.path.join(data_dir, "*.hdf5"))

        stem_name = str(Path(data_dir).stem)
        save_dir = os.path.join(_save_dir, stem_name)
        os.makedirs(save_dir, exist_ok=True)

        num_workers = len(datafiles)

        with Pool(num_workers) as pool:
            all_results = []
            for result in tqdm(pool.imap_unordered(process_file, datafiles), total=len(datafiles)):
                all_results.extend(result)
        
        random.shuffle(all_results)
        
        print(f"Loaded {len(all_results)} items from {data_dir}")

        chunk_id = 0
        for chunk in chunk_list_gen(all_results, chunk_size=500):
            parquet_file = os.path.join(save_dir, f"{stem_name}_{chunk_id}.parquet")
            save_to_parquet(chunk, parquet_file)
            chunk_id += 1

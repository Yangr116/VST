# Copyright 2025 [Visual Spatial Tuning] Authors
"""
1. read data, (data should be download and unzip)
2. inference using vllm
3. eval
"""

import os
from pathlib import Path
import yaml
from loguru import logger as eval_logger
from functools import partial
import numpy as np
import pandas as pd
import torch
import datasets
from datasets import DownloadConfig, Image, Sequence
import decord
import base64
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from vllm import LLM, SamplingParams
import copy
import collections
import json
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import argparse
from torch.utils.data import DataLoader, Dataset


MCA_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
]
NA_QUESTION_TYPES = [
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
]

METRICS_FOR_MCA = {
    "accuracy": "exact_match",
}

METRICS_FOR_NA = {
    "MRA:.5:.95:.05": "partial(mean_relative_accuracy, start=.5, end=.95, interval=.05)",
}


hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)

with open(Path(__file__).parent / "utils" / "vsibench.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
config = yaml.safe_load("".join(safe_data))
cache_name = config["dataset_kwargs"]["cache_dir"]


def vsibench_doc_to_visual(doc):
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = doc["dataset"] + "/" + doc["scene_name"] + ".mp4"
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    else:
        raise FileExistsError(f"video path:{video_path} does not exist.")
    return [video_path]


def vsibench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
        
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") or "These are frames of a video."
    
    if doc['question_type'] in NA_QUESTION_TYPES:
        post_prompt = lmms_eval_specific_kwargs.get("na_post_prompt", "") or "Please answer the question using a single word or phrase."
        return pre_prompt + "\n" + question + "\n" + post_prompt
    elif doc['question_type'] in MCA_QUESTION_TYPES:
        options = "Options:\n" + "\n".join(doc["options"])
        post_prompt = lmms_eval_specific_kwargs.get("mca_post_prompt", "") or "Answer with the option's letter from the given choices directly."
        return "\n".join([pre_prompt, question, options, post_prompt])
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    if os.getenv('LMMS_EVAL_SHUFFLE_DOCS', None):
        eval_logger.info(f"Environment variable LMMS_EVAL_SHUFFLE_DOCS detected, dataset will be shuffled.")
        return dataset.shuffle(seed=42)
    return dataset

def fuzzy_matching(pred):
    return pred.split(' ')[0].rstrip('.').strip()

def exact_match(pred, target):
    return 1. if pred.lower() == target.lower() else 0.

def abs_dist_norm(pred, target):
    return abs(pred - target) / target

def mean_relative_accuracy(pred, target, start, end, interval):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()

WORST_CASE_FOR_METRICS = {
    "accuracy": 0.,
    "MRA:.5:.95:.05": 0.,
}

def to_float(pred):
    try:
        pred = float(pred)
    except BaseException as e:
        pred = None
    return pred

def vsibench_process_results(doc, results):
    
    doc['prediction'] = results[0]
    if doc['question_type'] in MCA_QUESTION_TYPES:
        for key, value in METRICS_FOR_MCA.items():
            doc[key] = eval(value)(fuzzy_matching(doc['prediction']), doc['ground_truth'])
    elif doc['question_type'] in NA_QUESTION_TYPES:
        for key, value in METRICS_FOR_NA.items():
            try:
                doc[key] = eval(value)(to_float(fuzzy_matching(doc['prediction'])), to_float(doc['ground_truth']))
            except TypeError:
                doc[key] = WORST_CASE_FOR_METRICS[key]
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")

    return {"vsibench_score": doc}

def vsibench_aggregate_results(results):
    results = pd.DataFrame(results)
    
    output = {}

    for question_type, question_type_indexes in results.groupby('question_type').groups.items():
        per_question_type = results.iloc[question_type_indexes]
        
        if question_type in MCA_QUESTION_TYPES:
            for metric in METRICS_FOR_MCA.keys():
                output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
        elif question_type in NA_QUESTION_TYPES:
            for metric in METRICS_FOR_NA.keys():
                if metric == 'success_rate':
                    output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
                else:
                    output[f"{question_type}_{metric}"] = per_question_type[metric].mean()

        else:
            raise ValueError(f"Unknown question type: {question_type}")
    print(output)

    output['object_rel_direction_accuracy'] = sum([
        output.pop('object_rel_direction_easy_accuracy'),
        output.pop('object_rel_direction_medium_accuracy'),
        output.pop('object_rel_direction_hard_accuracy'),
    ]) / 3.
    
    output['overall'] = sum([_ for _ in output.values()]) / len(output)
    eval_logger.info(f"Evaluation results: {output}")
    return output['overall'] * 100.


class VSIDataset(Dataset):
    def __init__(self, dataset, processor, total_pixels=8192 * 28 * 28, fps=2, lmms_eval_specific_kwargs={}):
        super().__init__()
        self.dataset = dataset
        self.processor = processor
        self.total_pixels = total_pixels
        self.fps = fps
        self.lmms_eval_specific_kwargs = lmms_eval_specific_kwargs
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        sample = self.dataset[index]
        _sample = copy.deepcopy(sample)
        context = vsibench_doc_to_text(sample, self.lmms_eval_specific_kwargs)
        visuals = vsibench_doc_to_visual(sample)
        visual = visuals[0]
        if "<image>" in context:
            context = context.replace("<image>", "")
        _sample['context'] = context
        _sample['visual'] = visual
        message = [{"role": "system", "content": "You are a helpful assistant."}]
        if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
            # vr = decord.VideoReader(visual)
            # first_frame = vr[0].asnumpy()
            # height, width = first_frame.shape[:2]
            # max_pixels = height * width
            message.append({"role": "user", "content": [
                {"type": "video", "video": visual, "total_pixels": self.total_pixels, "min_pixels": 16 * 28 * 28, 'fps': self.fps}, 
                {"type": "text", "text": context}]})
        elif isinstance(visual, Image.Image):  # Single image
            base64_image = visual.convert("RGB")
            buffer = BytesIO()
            base64_image.save(buffer, format="JPEG")
            base64_bytes = base64.b64encode(buffer.getvalue())
            base64_string = base64_bytes.decode("utf-8")
            message.append({"role": "user", "content": [
                {"type": "image", "image": f"data:image/jpeg;base64,{base64_string}"}, 
                {"type": "text", "text": context}]})
        elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):  # Multiple images
            image_content = []
            for v in visual:
                base64_image = v.convert("RGB")
                buffer = BytesIO()
                base64_image.save(buffer, format="JPEG")
                base64_bytes = base64.b64encode(buffer.getvalue())
                base64_string = base64_bytes.decode("utf-8")
                image_content.append({"type": "image", "image": f"data:image/jpeg;base64,{base64_string}"})
            message.append({"role": "user", "content": image_content + [{"type": "text", "text": context}]})
        else:
            message.append({"role": "user", "content": [{"type": "text", "text": context}]})
        
        text = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(message, return_video_kwargs=True)

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        llm_inputs = {
            "prompt": text,
            "multi_modal_data": mm_data,
            # FPS will be returned in video_kwargs
            "mm_processor_kwargs": video_kwargs,
            "sample": _sample,
        }
        
        return llm_inputs


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    total_pixels = args.total_pixels
    fps = args.fps
    total_processor = args.total_processor # default to 1
    processor_id = args.processor_id

    # build dataset
    download_config = DownloadConfig()
    download_config.num_proc = 8
    download_config.local_files_only = False

    dataset = datasets.load_dataset(
        path="nyu-visionx/VSI-Bench",
        name=None,
        download_mode=datasets.DownloadMode.REUSE_DATASET_IF_EXISTS,
        download_config=download_config,
        token=True,
    )
    dataset = dataset['test']

    if total_processor > 1:
        len_dataset = len(dataset)
        chunk_size = len_dataset // total_processor
        start = chunk_size * processor_id
        end = start + chunk_size
        if processor_id == total_processor - 1:
            end = len_dataset
        dataset = datasets.load_dataset(
            path="nyu-visionx/VSI-Bench",
            name=None,
            split=f'test[{start}:{end}]',
            download_mode=datasets.DownloadMode.REUSE_DATASET_IF_EXISTS,
            download_config=download_config,
            token=True,)
        eval_logger.info(f"running {processor_id}/{total_processor}, {len(dataset)} samples")

    model_name_or_path = args.model_name_or_path
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    lmms_eval_specific_kwargs = config['lmms_eval_specific_kwargs']['default']
    vsidataset = VSIDataset(
        dataset=dataset, 
        processor=processor, 
        total_pixels=total_pixels, 
        fps=fps, 
        lmms_eval_specific_kwargs=lmms_eval_specific_kwargs)
    dataloader = DataLoader(
        dataset=vsidataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=lambda x: x,
    )

    # build model
    llm = LLM(
        model=model_name_or_path,
        gpu_memory_utilization=0.8,
        limit_mm_per_prompt={"image": 10, "video": 10},
        tensor_parallel_size=args.tp_size,
    )

    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.001,
        repetition_penalty=1.05,
        max_tokens=256,
        stop_token_ids=[],
    )

    results = []
    for idx, batch in enumerate(tqdm(dataloader)):
        # build context
        _results = []
        llm_inputs = []
        for _batch in batch:
            _results.append(_batch.pop('sample'))
            llm_inputs.append(_batch)
        
        outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
        for _sample, output in zip(_results, outputs):
            _sample['response'] = output.outputs[0].text
        results.extend(_results)

    # gather results
    task_outputs = collections.defaultdict(list)
    for result in results:
        res = vsibench_process_results(doc=result, results=[result['response']])
        for k, v in res.items():
            task_outputs[k].append(v)


    save_path = f"{args.output_dir}/results-{processor_id}-of-{total_processor}.json"
    eval_logger.info(f"save to {save_path}")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(task_outputs, f, indent=2, ensure_ascii=False)
    print(f"processor_id: {processor_id} done!")
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name_or_path", default="/opt/tiger/rayyang/spatial/cache/Qwen2-VL-2B-Instruct")
    parser.add_argument("--total-processor", default=1, type=int)
    parser.add_argument("--processor-id", default=0, type=int)
    parser.add_argument("--fps", default=4, type=int)
    parser.add_argument("--output-dir", default="work_dirs", type=str)
    parser.add_argument("--total-pixels", default=8192 * 28 * 28, type=int)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--tp-size", default=1, type=int)
    args = parser.parse_args()
    main(args)

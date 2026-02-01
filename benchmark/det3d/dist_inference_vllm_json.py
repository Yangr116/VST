import os
# Set environment variables before importing dependent libraries
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['NCCL_TIMEOUT'] = '1800'
os.environ['NCCL_BLOCKING_WAIT'] = '1'
os.environ['TORCH_NCCL_BLOCKING_WAIT_TIMEOUT'] = '1800'

import argparse
import copy
import json
import tempfile
from datetime import datetime
from io import BytesIO
from pathlib import Path

import base64
import cv2
import numpy as np
from loguru import logger
from PIL import Image
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader
from tqdm import tqdm

from types import SimpleNamespace
from vllm import LLM, SamplingParams

from vst.prerpocess_box3d import get_camera_params
from vst.utils.general import get_global_rank, save_json
from vst.utils.box3d_utils import extract_bbox, extract_bbox3d_from_json
from vst.utils.vis_utils import get_cuboid_verts_faces, draw_3d_box_from_verts

from benchmark.det3d.evaluation.evaluate import evaluate
from benchmark.det3d.dataset_val import ValDataset3DForVllmChat


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Qwen2VL model on a dataset.")
    # model args
    parser.add_argument("-m", "--model_name_or_path", type=str, required=True, help="Path to the model or model identifier from huggingface.co/models.")
    parser.add_argument("-bs", "--batch_size", default=1, type=int, help="batch_size")
    parser.add_argument("--max_images", default=32, type=int)
    parser.add_argument("--max_videos", default=8, type=int)
    parser.add_argument("--num_gpus", default=1, type=int)
    parser.add_argument("--outdir", type=str, required=True, help="Directory to save the evaluation results.")
    parser.add_argument("--dataset", type=str, nargs='+', required=True, help="Path to the dataset annotation file(s).")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the dataset.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing the images.")
    parser.add_argument("--gt-file", type=str, nargs='+', default=None, help="the ground truth file")
    parser.add_argument("--degree-range", type=str, default='180', help="the predicted degree range, support '1', '180', 'pi'")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--top_p", default=0.95, type=float)
    parser.add_argument("--enable_fov", default=False, action='store_true')
    parser.add_argument("--no_resize", default=False, action='store_true')
    parser.add_argument("--base_width", default=960, type=int)
    parser.add_argument("--enable_uvd", default=False, action='store_true')
    parser.add_argument("--enable_visualize", default=False, action='store_true')
    parser.add_argument("--enable_quat", default=False, action='store_true')
    parser.add_argument("--remove_intrinsics", default=False, action='store_true')
    parser.add_argument("--base_fov", default=69.16, type=float)
    args = parser.parse_args()
    if args.gt_file is None:
        args.gt_file = []
    args_dict = vars(args)
    for key, value in args_dict.items():
        print(f"{key}: {value}")
    return args


def build_data_loader(data_args: argparse.Namespace) -> DataLoader:
    """Build and return the data loader."""
    val_dataset = ValDataset3DForVllmChat(data_args)
    return DataLoader(
        dataset=val_dataset,
        batch_size=data_args.batch_size,
        num_workers=data_args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=lambda x: x,
        drop_last=False,
        shuffle=False
    )


def build_model_vllm(
        model_name_or_path: str, 
        tensor_parallel_size: int = 1, 
        gpu_memory_utilization: float = 0.9, 
        max_images: int = 32, 
        max_videos: int = 8,
        ):
    client = LLM(
        model=model_name_or_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        limit_mm_per_prompt={"image": max_images, "video": max_videos},
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=True,  # don't know why meet error when removing this line
    )
    return client


def unique_dicts_by_key(list_of_dicts, key='uid'):
    seen = set()
    unique_dicts = []
    
    for d in list_of_dicts:
        value = d[key]
        if value not in seen:
            unique_dicts.append(d)
            seen.add(value)
    
    return unique_dicts


def main():
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = "-".join([str(Path(name).stem) for name in args.dataset])

    outdir = os.path.join(args.outdir, f'{timestamp}_{dataset_name}_results')
    os.makedirs(outdir, exist_ok=True)

    # model
    logger.info("Loading model...")
    vllm_args = dict(
        model_name_or_path=args.model_name_or_path, 
        tensor_parallel_size=args.num_gpus, 
        gpu_memory_utilization=0.5, 
        max_images=args.max_images, 
        max_videos=args.max_videos,
    )
    model = build_model_vllm(**vllm_args)

    params = {
        "temperature": args.temperature,
        "max_tokens": args.max_new_tokens,
        "top_p": args.top_p,
        "skip_special_tokens": False,
    }
    sampling_params = SamplingParams(**params)

    for idx, ann_path in enumerate(args.dataset):
        # data
        data_args = argparse.Namespace(
            ann_path=os.path.join(args.data_dir, ann_path),
            image_dir=os.path.join(args.data_dir, args.image_dir),
            batch_size=args.batch_size,
            num_workers=4,
            enable_fov=args.enable_fov,
            no_resize=args.no_resize,
            base_width=args.base_width,
            enable_uvd=args.enable_uvd,
            enable_quat=args.enable_quat,
            remove_intrinsics=args.remove_intrinsics,
            base_fov=args.base_fov,
        )
        rotation_type = 'quat' if args.enable_quat else 'euler'
        if len(args.gt_file) > idx:
            gtfile = os.path.join(args.data_dir, args.gt_file[idx])
        else:
            gtfile = None
        dataloader = build_data_loader(data_args)
        savefile = os.path.join(outdir, f'predictions_{str(Path(ann_path).stem)}.json')
        
        results = []
        for i, batch in enumerate(dataloader):
            _results = []
            batched_messages = []
            for (messages, meta_info) in batch:
                _results.append(copy.deepcopy(meta_info))
                batched_messages.append(messages)

            response = model.chat(sampling_params=sampling_params, messages=batched_messages)
            
            for j, (output_text, _result) in enumerate(zip(response, _results)):
                _result['predictions'] = extract_bbox3d_from_json(output_text.outputs[0].text, from_range=args.degree_range, to_range='180', rotation_type=rotation_type)
                _result['predictions_raw'] = output_text.outputs[0].text

                if args.enable_visualize:
                    visualize_results(batch[j][0], _result['predictions'], _result, i+j)
            results.extend(_results)
            if i == 0:
                logger.info(f"sample result:\n {json.dumps(results[0], indent=2)}")
        # save results into split json
        savefile_rank = os.path.join(str(Path(savefile).parent), f"{Path(savefile).stem}_{get_global_rank()}.json")
        save_json(results, save_path=savefile_rank)

        # all results may have repeated instances because of ddp, remove them according to uid
        all_results = results
        # all_results = unique_dicts_by_key(all_results)
        save_json(all_results, save_path=savefile)
        
        if gtfile is not None:
            for i in range(3):
                try:
                    evaluate(gtfile=gtfile, pred_file=savefile, outdir=outdir)
                    break
                except Exception as e:
                    logger.error(e)
    
    logger.success("Done!")


def visualize_results(message, predictions, ori_prompt, image_name):
    camera_params = get_camera_params(ori_prompt['original_question'])
    original_w, original_h, fx, fy, cx, cy = camera_params

    K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]

    image = None
    for x in message:
        for content in x['content']:
            if content['type'] == 'image_url':
                base64_image_data = content['image_url']['url']
                if "," in base64_image_data:
                    base64_image_data = base64_image_data.split(",")[1]
                image_data = base64.b64decode(base64_image_data)
                image = Image.open(BytesIO(image_data))
                print(image.size)
                image = image.resize((int(original_w), int(original_h)))
                image = np.array(image)

    objects = []
    for box_ann in predictions:
        euler_angle_pyr = box_ann[-3:]
        rotation = Rotation.from_euler(angles=euler_angle_pyr, seq='xyz', degrees=True)
        R_bbox2cam = rotation.as_matrix()
        objects.append(
            SimpleNamespace(centroid_cam=box_ann[0:3], 
                dimensions=box_ann[3:6], R_bbox2cam=R_bbox2cam.tolist(),
                bbox2D_tight=None)
        )
    for obj in objects:
        box3d = obj.centroid_cam + obj.dimensions
        R = np.array(obj.R_bbox2cam)
        verts3d, _ = get_cuboid_verts_faces(box3d=box3d, R=R)
        # Convert to pixel coordinates
        draw_3d_box_from_verts(im=image, K=K, verts3d=verts3d, draw_back=True, draw_top=True)
    cv2.imwrite(f'{image_name}.jpg', image)


if __name__ == "__main__":
    main()

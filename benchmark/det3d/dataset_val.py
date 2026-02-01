import os
import math
import base64
from PIL import Image
from io import BytesIO
from copy import deepcopy
from typing import Union
from loguru import logger
from torch.utils.data import Dataset
from qwen_vl_utils import smart_resize

from vst.utils.general import load_json
from vst.prompt import SYS_PROMPT
from vst.prerpocess_box3d import convert_prefix_prompt_to_fov, xyz_prompt, uvd_prompt, quat_prompt, func_remove_intrinsics


def encode_image_to_base64(image: Union[Image.Image, str], resized_height=None, resized_width=None):
    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
    else:
        img = image.copy()

    if resized_height is not None and resized_width is not None:
        resized_height, resized_width = smart_resize(
            resized_height,
            resized_width,
            factor=28,
        )
        img = img.resize((resized_width, resized_height))

    output_buffer = BytesIO()
    img.save(output_buffer, format="PNG")
    byte_data = output_buffer.getvalue()

    base64_str = base64.b64encode(byte_data).decode("utf-8")
    return base64_str


class ValDataset3D(Dataset):
    def __init__(self, data_args):
        super().__init__()
        logger.info(f"Loading annotation from: {data_args.ann_path}")
        self.image_dir = data_args.image_dir
        self.json_data = load_json(data_args.ann_path)
        logger.info(f"{data_args.ann_path} @ samples: {len(self.json_data)}")
        logger.info(f"sample:\n{convert_qa2qwen_val(self.json_data[0])}")

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, idx):
        item = self.json_data[idx]
        item["image_path"] = os.path.join(self.image_dir, item["image_path"])
        messages = convert_qa2qwen_val(item)
        return (messages, item)


def convert_qa2qwen_val(sample):
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYS_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": sample["image_path"],
                },
                {
                    "type": "text",
                    "text": sample["question"],
                },
            ],
        }
    ]


class ValDataset3DForVllmChat(Dataset):
    def __init__(self, data_args):
        super().__init__()
        logger.info(f"Loading annotation from: {data_args.ann_path}")
        self.image_dir = data_args.image_dir
        self.json_data = load_json(data_args.ann_path)
        self.enable_fov = data_args.enable_fov
        self.no_resize = data_args.no_resize
        self.base_width = data_args.base_width
        self.base_fov = data_args.base_fov
        self.base_focal = self.base_width / (2 * math.tan(math.radians(self.base_fov) / 2))
        self.enable_uvd = data_args.enable_uvd
        self.enable_quat = data_args.enable_quat
        self.remove_intrinsics = getattr(data_args, 'remove_intrinsics', False)
        logger.info(f"{data_args.ann_path} @ samples: {len(self.json_data)}")
        logger.info(f"sample:\n{self.json_data[0]}")

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, idx):
        item = deepcopy(self.json_data[idx])
        if isinstance(item["image_path"], list):
            item['image_path'] = [os.path.join(self.image_dir, image_path) for image_path in item["image_path"]]
        else:
            item['image_path'] = os.path.join(self.image_dir, item["image_path"])
            # image = Image.open(item['image_path'])
            # print(f"image.size: {image.size}")
        item['image_resized_height'] = None
        item['image_resized_width'] = None
        item['original_question'] = deepcopy(item['question'])
        if self.enable_fov:
            new_prompt, new_size, camera_params = convert_prefix_prompt_to_fov(item['question'], new_focal_length=self.base_focal, no_resize=self.no_resize)
            # print(f"new_size: {new_size}")
            if new_size is not None:
                item['question'] = new_prompt
                item['image_resized_height'] = new_size[1]
                item['image_resized_width'] = new_size[0]
            else:
                item['image_resized_height'] = None
                item['image_resized_width'] = None
        if self.enable_uvd:
            item['question'] = item['question'].replace(xyz_prompt, uvd_prompt)
        if self.enable_quat:
            if "question_original" not in item:
                item['question_original'] = deepcopy(item['question']) 
            item['question'] = item['question'].replace(xyz_prompt, quat_prompt)
        if self.remove_intrinsics:
            item['question'] = func_remove_intrinsics([{'value': item['question']}])[0]['value']

        # logger.info("==="*10 + f"self.json_data[idx]: {self.json_data[idx]}\nitem['question']: {item['question']}")
        messages = convert_qa2vllm(item)
        return (messages, item)


def convert_qa2vllm(sample):
    if isinstance(sample["image_path"], list):
        base64_image_list = [encode_image_to_base64(image_path, resized_height=sample["image_resized_height"], resized_width=sample["image_resized_width"]) for image_path in sample["image_path"]]
    else:
        base64_image_list = [encode_image_to_base64(sample["image_path"], resized_height=sample["image_resized_height"], resized_width=sample["image_resized_width"])]
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYS_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}} for base64_image in base64_image_list]
                + [{
                    "type": "text",
                    "text": sample["question"],
                }],
        }
    ]

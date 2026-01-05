# Copyright 2025 [Visual Spatial Tuning] Authors
import torch
import copy
import math
from PIL import Image
from io import BytesIO
from typing import Any, Dict, Callable, Optional
from transformers import ProcessorMixin
from loguru import logger

from veomni.data.constants import IMAGE_INPUT_INDEX, VIDEO_INPUT_INDEX
from veomni.data.chat_template import ChatTemplate

from vst.prompt import SYS_PROMPT, THINK_SYSTEM_PROMPT
from vst.prerpocess_box3d import convert_prefix_prompt_to_fov, convert_bbox3d_to_uvd, convert_bbox3d_to_quat, func_remove_intrinsics, convert_conv_predict_fov
from vst.utils.vision_process import process_vision_info, VIDEO_MIN_PIXELS


IMAGE_FLAG = "<|image_pad|>"
VIDEO_FLAG = "<|video_pad|>"

# 1. we convert the llave format data into qwen messages, this helps to recognize image flag
# 2. we convert the qwen messages into constructed_conversation

def convert_llava2qwen(sample, include_system=True, total_pixels=int(16384*28*28*0.9), fps=4):
    role_mapping = {"human": "user", "gpt": "assistant"}
    result = []
    if include_system:
        if 'thought' in sample.get('type', 'unknown'):
            result.append({
                "role": "system",
                "content": [{"type": "text", "text": THINK_SYSTEM_PROMPT}],
            })
        else:
            result.append({
                "role": "system",
                "content": [{"type": "text", "text": SYS_PROMPT}],
            })
    
    source = sample["conversations"] if "conversations" in sample else sample["text"]  # text-only data

    if source[0]['from'] != "human":
        logger.warning(f"The first role of this sample isn't from human {sample}")
        source = source[1:]

    image_id = 0
    # get all images
    images: list = sample.get("images")
    if 'image_resized_height' in sample and 'image_resized_width' in sample:
        resized_dict = {"resized_height": sample['image_resized_height'], "resized_width": sample['image_resized_width']}
    elif len(images) > 1:
        # NOTE: For multiple images, we limit their total pixels
        max_pixels = max(total_pixels / len(images), int(VIDEO_MIN_PIXELS * 1.05))
        resized_dict = {"max_pixels": max_pixels}
    else:
        resized_dict = {}
    
    for idx, conv in enumerate(source):
        if 'from' not in conv or 'value' not in conv:
            continue
        # skip empty
        if not conv.get('value', '').strip():
            continue
        role = role_mapping.get(conv['from'])
        if not role:
            continue

        value = conv['value']
        if role == "user":
            content = []
            if idx == 0 and len(images) > 0 and IMAGE_FLAG not in value and VIDEO_FLAG not in value:
                # remove all image flag
                if "<image>" in value:
                    value = value.replace("<image>\n", "").replace("\n<image>", "").replace("<image>", "")
                value = f"{IMAGE_FLAG}"*len(images) + value

            if VIDEO_FLAG in value:
                assert len(images) > 0
                split_values = value.split(VIDEO_FLAG)
                content = []
                for i in range(len(split_values)):
                    split_value = split_values[i].strip()
                    if split_value != '':
                        content.append({'type': 'text', 'text': split_value})
                    if i < len(split_values) - 1:
                        content.append({'type': 'video', 'video': images[image_id], 'fps': fps, 'total_pixels': total_pixels})  # we can set the fps here
                        image_id += 1
            elif IMAGE_FLAG in value:
                assert len(images) > 0
                split_values = value.split(IMAGE_FLAG)
                content = []
                for i in range(len(split_values)):
                    split_value = split_values[i].strip()
                    if split_value != '':
                        content.append({'type': 'text', 'text': split_value})
                    if i < len(split_values) - 1:
                        content.append({'type': 'image', 'image': images[image_id], **resized_dict})
                        image_id += 1
            else:
                content = [{'type': 'text', 'text': value}]
        else:
            content = [{'type': 'text', 'text': value}]
        result.append({"role": role, "content": content})
    return result


def qwen_messages_sft_preprocess(messages, **kwargs):
    """
    Return:
        list[list]. Example: [[role, ('text', xxx)], ...]
    """
    constructed_conversation = []
    for message in messages:
        _conv = [message['role']]
        for _content in message['content']:
            content_type = _content['type']
            _conv_item = (content_type, _content[content_type]) if content_type == "text" else (content_type, None)
            _conv.append(_conv_item)
        constructed_conversation.append(_conv)
    return constructed_conversation


def llava_sft_preprocess(conversations, **kwargs):
    role_mapping = {"human": "user", "gpt": "assistant"}
    constructed_conversation = []
    if conversations[0]["from"] != "human":  # Skip the first one if it is not from human
        conversations = conversations[1:]
    assert conversations[0]["from"] == "human"

    for message in conversations:
        value = message["value"]
        role = role_mapping[message["from"]]
        if value is None:
            constructed_conversation.append(None)  # eval
        else:
            if "<image>" in value:
                value = value.replace("<image>\n", "").replace("\n<image>", "").replace("<image>", "")
                constructed_conversation.append([role, ("image", None), ("text", value)])
            else:
                constructed_conversation.append([role, ("text", value)])
    return constructed_conversation


def load_vision_inputs(sample):
    images = []
    data_type = sample.get('type', 'unknown')
    if 'video' in data_type:
        for vision_item in sample["images"]:
            images.append(BytesIO(vision_item['bytes']))
    elif data_type != 'text':
        if "images" in sample:
            if sample["images"] is None:
                sample["images"] = []
            for image in sample["images"]:
                if isinstance(image, dict):
                    images.append(Image.open(BytesIO(image['bytes'])).convert("RGB"))
                else:
                    images.append(Image.open(BytesIO(image)).convert("RGB"))
        elif "image" in sample:
            image = sample["image"]
            if image is None:
                pass
            elif isinstance(image, dict):
                images.append(Image.open(BytesIO(image['bytes'])).convert("RGB"))
            elif isinstance(image, Image.Image):
                images.append(image)
            else:
                raise NotImplementedError
    return images


class SampleTransform:
    def __init__(
            self, 
            processor: "ProcessorMixin", 
            chat_template: "ChatTemplate", 
            position_id_func: "Callable",
            enable_fov: bool = False,
            enable_uvd: bool = False,
            include_system: bool = True,
            no_resize: bool = False,
            enable_quat: bool = False,
            remove_intrinsics: bool = False,
            base_focal: Optional[float] = None,
            max_seq_len: int = 8192,
            fps=2,
            is_warmup=False,
            enable_predict_fov=False,
            **kwargs):
        self.processor = processor
        self.chat_template = chat_template
        self.position_id_func = position_id_func
        self.enable_fov = enable_fov
        self.enable_uvd = enable_uvd
        self.include_system = include_system
        self.no_resize = no_resize
        self.enable_quat = enable_quat
        self.remove_intrinsics = remove_intrinsics
        self.max_seq_len = max_seq_len
        logger.info(f"set max_seq_len to {max_seq_len} in processor")
        self.base_focal = base_focal if base_focal is not None else 960 / (2 * math.tan(math.radians(69.16) / 2))
        logger.info(f"set base focal to {self.base_focal}")
        self.total_pixels = int(max_seq_len * 28 * 28 * 0.8)  # video total pixels
        self.fps = fps  # video fps
        logger.info(f"set video parameters total_pixels={self.total_pixels} fps={self.fps}")
        self.is_warmup = is_warmup
        self.enable_predict_fov = enable_predict_fov

        # set model image_id and video_id, this is align with the model


    def process_bbox3d(self, _sample: Dict):
        sample = copy.deepcopy(_sample)
        camera_params = None
        new_size = None
        if self.enable_fov:
            new_focal_length = self.base_focal
            ori_size = sample['images'][0].size # w, h
            new_prompt, new_size, camera_params = convert_prefix_prompt_to_fov(
                copy.deepcopy(sample['conversations'][0]['value']),
                ori_size=ori_size,
                new_focal_length=new_focal_length,
                no_resize=self.no_resize)
            if new_size is not None:
                sample['conversations'][0]['value'] = new_prompt
                sample['image_resized_height'] = new_size[1]
                sample['image_resized_width'] = new_size[0]
            # resize here
        elif self.enable_predict_fov:
            sample['conversations'] = convert_conv_predict_fov(copy.deepcopy(sample['conversations']))
        
        if self.enable_uvd:
            sample['conversations'] = convert_bbox3d_to_uvd(copy.deepcopy(sample['conversations']), camera_params, new_size)  
            # NOTE: 这里的实现有问题，new size 不一定会被 28 整除，但是 gt 都是按照这种方式来的

        if self.enable_quat:
            sample['conversations'] = convert_bbox3d_to_quat(copy.deepcopy(sample['conversations']), camera_params, new_size)
        
        if self.remove_intrinsics:
            sample['conversations'] = func_remove_intrinsics(copy.deepcopy(sample['conversations']))
        return sample

    def __call__(self, sample, **kwargs):
        """
        Processes multimodal example with qwen2vl's pre-processor.
        """
        # preprocess the original data, e.g., remove the "<image>" flag from data
        # conversations = sample["conversations"] if "conversations" in sample else sample["text"]  # text-only data
        # load images
        if 'text' in sample.get('type', 'unknown'):
            sample['images'] = []
        else:
            sample['images'] = load_vision_inputs(sample)
        
        if sample.get('type', 'unknown') == "bbox_3d":
            sample = self.process_bbox3d(sample)
        
        if self.is_warmup and len(sample['images']) > 0:
            sample['conversations'][0]['value'] = f"{IMAGE_FLAG}"*len(sample['images'])

        messages = convert_llava2qwen(sample, self.include_system, total_pixels=self.total_pixels, fps=self.fps)
        # add size into conversations and get images
        image_inputs, video_inputs = process_vision_info(messages) # resize image if needed
        
        messages = qwen_messages_sft_preprocess(messages, **kwargs)

        token_num_inputs, vision_inputs = {}, {}
        image_grid_thw = None
        video_grid_thw = None
        merge_length = self.processor.image_processor.merge_size**2

        # We only support video or image
        if video_inputs is not None:
            vision_inputs = self.processor.image_processor(images=None, videos=video_inputs, return_tensors="pt")
            video_grid_thw = vision_inputs["video_grid_thw"]
            video_token_num = video_grid_thw.prod(dim=-1) // merge_length
            token_num_inputs["video"] = video_token_num
        elif image_inputs is not None:
            vision_inputs = self.processor.image_processor(images=image_inputs, return_tensors="pt")
            image_grid_thw = vision_inputs["image_grid_thw"]
            image_token_num = image_grid_thw.prod(dim=-1) // merge_length # this is a list
            token_num_inputs["image"] = image_token_num
        
        tokenized_example = self.chat_template.encode_messages(messages, token_num_inputs)
        tokenized_example = {k: v if isinstance(v, torch.Tensor) else torch.tensor(v) for k, v in tokenized_example.items()}
        input_ids = tokenized_example["input_ids"]

        if input_ids.shape[0] > self.max_seq_len:
            input_ids[input_ids < 0] = 151655
            raise ValueError(f"skip, {input_ids.shape[0]} > {self.max_seq_len}\n{self.chat_template.tokenizer.decode(input_ids)}")

        position_ids = self.position_id_func(
            input_ids=input_ids.unsqueeze(0),
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=tokenized_example["attention_mask"].unsqueeze(0),
        )["position_ids"]
        if position_ids is not None:
            tokenized_example["position_ids"] = position_ids.squeeze().clone()  # (dim, l)

        tokenized_example["image_mask"] = tokenized_example["input_ids"] == IMAGE_INPUT_INDEX
        tokenized_example["video_mask"] = tokenized_example["input_ids"] == VIDEO_INPUT_INDEX
        tokenized_example["input_ids"][tokenized_example["image_mask"]] = 0
        tokenized_example["input_ids"][tokenized_example["video_mask"]] = 0
        tokenized_example.update(vision_inputs)
        return [tokenized_example]

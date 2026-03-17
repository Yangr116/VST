# Copyright 2025 [Visual Spatial Tuning] Authors
import json
import torch
from typing import Any, Dict, Callable, Optional
from transformers import ProcessorMixin
import torchvision.transforms as T
from loguru import logger

from veomni.data.constants import IMAGE_INPUT_INDEX, VIDEO_INPUT_INDEX
from veomni.data.chat_template import ChatTemplate

from vst.preprocess import load_vision_inputs, convert_llava2qwen, qwen_messages_sft_preprocess
from vst.utils.vision_process import process_vision_info
from vst.vla.action_tokenizer import ActionTokenizer


class SampleTransformVLA:
    def __init__(
            self, 
            processor: "ProcessorMixin", 
            chat_template: "ChatTemplate", 
            position_id_func: "Callable",
            action_tokenizer: "ActionTokenizer",
            max_seq_len: int = 8192,
            enable_augmentation: bool = True,
            image_size: int = 256,
            **kwargs,
    ) -> None:
        self.processor = processor
        self.chat_template = chat_template
        self.position_id_func = position_id_func
        self.action_tokenizer = action_tokenizer
        self.max_seq_len = max_seq_len
        self.enable_augmentation = enable_augmentation


        H, W = image_size, image_size
        logger.warning(f"We set the image_size to {image_size}")
        scale = (0.8, 1.0)
        ratio = (1.0, 1.0)
        self.image_aug = T.Compose([
            T.RandomResizedCrop(size=(H, W), scale=scale, ratio=ratio),
            T.ColorJitter(
                brightness=0.2,
                contrast=(0.8, 1.2),
                saturation=(0.8, 1.2),
                hue=0.05
            ),
        ])

    def __call__(self, sample: Dict, **kwargs) -> Dict:
        sample['images'] = load_vision_inputs(sample)
        meta_info = json.loads(sample['meta_info'])
        action, language_instruction = meta_info['action'], meta_info['language_instruction']
        human_value = f"What action should the robot take to {language_instruction}?"
        gpt_value = self.action_tokenizer(action)
        conversations = [
            {'from': 'human', 'value': human_value},
            {'from': 'gpt', 'value': gpt_value},
        ]
        sample['conversations'] = conversations
        messages = convert_llava2qwen(sample, include_system=False)
        image_inputs, video_inputs = process_vision_info(messages)
        if self.enable_augmentation and image_inputs is not None and random.random() > 0.5:
            new_image_inputs = []
            for image_input in image_inputs:
                new_image_inputs.append(self.image_aug(image_input))
            image_inputs = new_image_inputs

        messages = qwen_messages_sft_preprocess(messages, **kwargs)

        token_num_inputs, vision_inputs = {}, {}
        image_grid_thw = None
        video_grid_thw = None
        merge_length = self.processor.image_processor.merge_size**2

        if image_inputs is not None:
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
        tokenized_example["position_ids"] = position_ids  # (dim, l)

        tokenized_example["image_mask"] = tokenized_example["input_ids"] == IMAGE_INPUT_INDEX
        tokenized_example["video_mask"] = tokenized_example["input_ids"] == VIDEO_INPUT_INDEX
        tokenized_example["input_ids"][tokenized_example["image_mask"]] = 0
        tokenized_example["input_ids"][tokenized_example["video_mask"]] = 0
        tokenized_example.update(vision_inputs)
        return [tokenized_example]

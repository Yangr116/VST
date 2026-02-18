# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/registry.py

import importlib
import pkgutil
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Type, Union

import torch.nn as nn

from veomni.utils import logging
from veomni.models.registry import ModelRegistry

from vst.models.transformers.qwen2_5vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from vst.models.transformers.qwen3vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
logger = logging.get_logger(__name__)


logger.warning(f"="*10+ "Register new models" +"="*10)

models = [
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration
]

for model in models:
    if model.__name__ in ModelRegistry.model_arch_name_to_cls:
        logger.warning(f"Replace {ModelRegistry.model_arch_name_to_cls[model.__name__]} into {model}")
        ModelRegistry.model_arch_name_to_cls[model.__name__] = model
    else:
        logger.warning(f"Register {model}")
        ModelRegistry.model_arch_name_to_cls[model.__name__] = model

logger.info(f"supported models: {ModelRegistry.supported_models}")

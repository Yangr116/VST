# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import re
import torch
from typing import Dict

from mathruler.grader import grade_answer
from examples.reward_function.box3d_utils import extract_bbox3d_from_json, compute_box3d_from_answer, box3d_overlap, convert_degree_range, parse_json
from scipy.optimize import linear_sum_assignment
import numpy as np
from loguru import logger
import json
import copy


def format_reward(predict_str: str) -> float:
    """json + each entry has bbox_3d and label as key"""
    try:
        output_string = parse_json(predict_str)
        results = json.loads(output_string)
        score = 1.0
        for x in results:
            if not "bbox_3d" in x or not "label" in x:
                score = 0
                break
            elif len(x["bbox_3d"]) != 9:
                score = 0
                break
        return score
    except Exception as e:
        logger.error(f"Error: {e}")
        return 0


def get_vertices_from_answer(answer):
    if isinstance(answer, str):
        bbox3d_list = extract_bbox3d_from_json(answer, from_range='1')
    else:
        bbox3d_list = [convert_degree_range(x['bbox_3d'], from_range='1', to_range='180') for x in answer]
    vertices_list = []
    for bbox3d in bbox3d_list:
        vertices = compute_box3d_from_answer(bbox3d)
        vertices_list.append(vertices)
    return torch.stack(vertices_list)


def iou3d_reward(predict_str: str, ground_truth: list) -> float:

    try:
        vertices_pred = get_vertices_from_answer(predict_str)
        vertices_gt = get_vertices_from_answer(copy.deepcopy(ground_truth))
        # iou
        ious = box3d_overlap(vertices_pred, vertices_gt).cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(-ious)
        # iou = np.mean([ious[r, c] for r, c in zip(row_ind, col_ind)])
        tp = 0
        matched_gt = set()
        for r, c in zip(row_ind, col_ind):
            if ious[r, c] >= 0.25:
                tp += 1
                matched_gt.add(c)
        
        fp = len(vertices_pred) - tp
        fn = len(vertices_gt) - len(matched_gt)
        
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
        iou_mean = np.sum([ious[r, c] for r, c in zip(row_ind, col_ind)]) / len(vertices_gt)
        score = 0.5 * iou_mean + 0.5 * f1
        return score
    except Exception as e:
        logger.error(f"Error: {e}\npredict_str:\n{predict_str}, ground_truth:\n{ground_truth}")
        return 0.


def compute_score(predict_str: str, ground_truth: str, format_weight: float = 0.5) -> Dict[str, float]:
    format_score = format_reward(predict_str)
    accuracy_score = iou3d_reward(predict_str, ground_truth)
    return {
        "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
        "format": format_score,
        "accuracy": accuracy_score,
    }


if __name__ == "__main__":
    predict_str = "```json\n[\n\t{\"bbox_3d\":[0.7,-0.5,2.46,0.6,0.9,0.56,0.61,0.33,0.6],\"label\":\"chair\"}\n]\n```"
    ground_truth = [{"bbox_3d":[0.34,0.09,1.64,0.54,0.89,0.51,0.63,0.31,0.63],"label":"chair"},{"bbox_3d":[-0.28,0.05,1.77,0.53,0.87,0.52,0.64,0.31,0.64],"label":"chair"},{"bbox_3d":[1.4,-0.41,2.34,0.55,0.87,0.49,0.63,0.31,0.63],"label":"chair"},{"bbox_3d":[0.76,-0.47,2.51,0.58,0.86,0.54,0.65,0.31,0.65],"label":"chair"},{"bbox_3d":[0.02,-0.49,2.66,0.54,0.9,0.51,0.63,0.31,0.63],"label":"chair"}]
    score = compute_score(predict_str, ground_truth)
    print(score)

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
import json
from typing import Any, Dict, List

from mathruler.grader import grade_answer
from examples.reward_function.reward_func import get_accuracy_reward
from examples.reward_function.iou_reward import compute_score as compute_score_bbox3d


THINK_TAG_PATTERN = re.compile(r"<think>.*?</think>", flags=re.DOTALL)

def format_reward(response: str) -> float:
    matches = THINK_TAG_PATTERN.findall(response)
    if len(matches) != 1:
        return 0.0
    answer = response.split('</think>')[-1].strip()
    if answer == "":
        return 0.0
    return 1.0


def extract_option(pred):
    pattern = r'\b[A-D]\b(?!\s[a-zA-Z])'
    match = re.search(pattern, pred)
    if match:
        pred = match.group()  # 提取孤立的大写字母（排除"A bike"，不定冠词+空格+单词的情况）
    else:
        return None
    return pred


def accuracy_reward(response: str, ground_truth: str, reward_mode: List) -> float:
    try:
        # content_match = re.search(r"<answer>(.*?)</answer>", response)
        # given_answer = content_match.group(1).strip() if content_match else response.strip()
        given_answer = response.split('</think>')[-1].strip()
        # if extract_option(given_answer).lower() == extract_option(ground_truth.strip()).lower():
        #     return 1.0
        if '\\boxed' in ground_truth:
            pattern = r'\\boxed\{(.*?)\}'
            match_pred = re.search(pattern, given_answer)
            if match_pred:
                given_answer = match_pred.group(1)
            
            match_gt = re.search(pattern, ground_truth)
            if match_pred:
                ground_truth = match_gt.group(1)

        acc = get_accuracy_reward(answer_pred=given_answer, answer_gt=ground_truth, mode=reward_mode)
        return acc

    except Exception:
        pass

    return 0.0


def compute_score(reward_input: Dict[str, Any], format_weight: float = 0.5) -> Dict[str, float]:
    if not isinstance(reward_input, dict):
        raise ValueError("Please use `reward_type=sequential` for r1v reward function.")
    
    prediction = reward_input["response"]
    ground_truth = reward_input["ground_truth"]
    reward_mode = json.loads(reward_input['reward_mode'])

    if 'bbox3d' in reward_mode:
        return compute_score_bbox3d(predict_str=prediction, ground_truth=ground_truth, format_weight=format_weight)

    format_score = format_reward(prediction)
    accuracy_score = accuracy_reward(prediction, ground_truth, reward_mode=reward_mode)
    return {
        "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
        "format": format_score,
        "accuracy": accuracy_score,
    }


if __name__ == "__main__":
    response = """<think>Step 1: Identify common objects to analyze camera motion. Both images feature desks, chairs, and a large whiteboard. The first image shows desks in the foreground and chairs in the midground, while the second image has desks and chairs in the midground and background, with new elements like a door, clock, and ceiling lights.
(Runner pid=83491) Step 2: Deduce camera movement type. The change in visible elements (from foreground desks to midground desks, new background features) indicates a pan (rotation around the vertical axis). For a pan, the camera's motion is a rotation around its vertical axis. 
(Runner pid=83491) Step 3: Determine the pan direction. The second image's camera is pointed at 9 o'clock. To find the first image's direction, reverse the pan motion. A pan to the right (camera turning to the right) means the first image's camera was to the left of the second image's camera. The 9 o'clock direction corresponds to the right side of the scene. So, the first image's camera direction is left of 9 o'clock.
(Runner pid=83491) Step 4: Evaluate options. Among the options, 8 o'clock is the only direction left of 9 o'clock. Other options (2 o'clock, 5 o'clock, 6 o'clock) are either right of 9 o'clock (2 o'clock, 5 o'clock) or far left (6 o'clock). Thus, the first image's camera direction is 8 o'clock.</think> A. 8 o'clock"""
    acc_reward = accuracy_reward(response, ground_truth="A. 8 o'clock")
    f_reward = format_reward(response)
    print(f"acc_reward: {acc_reward}, f_reward: {f_reward}")

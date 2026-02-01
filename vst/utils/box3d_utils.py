# Copyright 2025 [Visual Spatial Tuning] Authors

import re
import json
import math
from loguru import logger
from typing import List, Dict


def extract_bbox(input_string, from_range: str='180', to_range: str = '180'):
    def _extract_bbox(input_string, from_range: str='180', to_range: str = '180'):
        """Extract 3D bounding boxes from the input string."""
        results = []
        matches = re.findall(r'<3dbbox>(.*?)</3dbbox>', input_string)
        for bbox3d in matches:
            bbox3d = bbox3d.split(' ')
            bbox3d = list(map(float, bbox3d))
            if len(bbox3d) != 9: continue
            bbox3d = convert_degree_range(bbox3d, from_range=from_range, to_range=to_range)
            results.append(bbox3d)
        return results
    try:
        bboxes3d = _extract_bbox(input_string, from_range=from_range, to_range=to_range)
    except Exception as e:
        logger.error(f"Error processing sample: {input_string}, Error: {e}")
        bboxes3d = []
    return bboxes3d


def parse_json(json_output: str):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output


def decode_bbox3d_dict_from_json(input_string):
    output_string = parse_json(input_string)
    results = json.loads(output_string)
    return results


def encode_json(data: List[Dict]):
    lines = [f'\t{json.dumps(item)}' for item in data]
    formatted_lines = ',\n'.join(lines)
    array_str = f'[\n{formatted_lines}\n]'
    result = f'```json\n{array_str}\n```'
    return result


def extract_bbox3d_from_json(input_string, from_range: str='180', to_range: str = '180', rotation_type='euler'):
    def _extract_bbox(input_string, from_range: str='180', to_range: str = '180', rotation_type: str = 'euler'):
        output_string = parse_json(input_string)
        results = json.loads(output_string)
        bboxes3d = [x.get("bbox_3d", None) for x in results]
        if rotation_type == 'quat':
            bboxes3d = [convert_quat2euler(x) for x in bboxes3d if x is not None and len(x) == 10]
        else:
            bboxes3d = [convert_degree_range(x, from_range=from_range, to_range=to_range) for x in bboxes3d if x is not None and len(x)==9]
        return bboxes3d
    try:
        bboxes3d = _extract_bbox(input_string, from_range=from_range, to_range=to_range, rotation_type=rotation_type)
    except:
        bboxes3d = []
    return bboxes3d


def convert_quat2euler(bbox3d: list):
    rotation = Rotation.from_quat(bbox3d[-4:])
    euler = rotation.as_euler('xyz', degrees=True)
    bbox3d = bbox3d[:6] + euler.tolist()
    return bbox3d


def convert_degree_range(bbox3d: list, from_range: str='180', to_range: str = '180'):
    """
        from_range: support '1', 'pi', '180'
        to_range: support '1', 'pi', '180'
    """
    if from_range == to_range:
        return bbox3d
    if from_range == '1' and to_range == '180':
        bbox3d[-3:] = map(lambda x: x*180, bbox3d[-3:])
    elif from_range == 'pi' and to_range == '180':
        bbox3d[-3:] = map(lambda x: x*180 / math.pi, bbox3d[-3:])
    elif from_range == '180' and to_range == '1':
        bbox3d[-3:] = map(lambda x: x / 180, bbox3d[-3:])
    elif from_range == '180' and to_range == 'pi':
        bbox3d[-3:] = map(lambda x: x*math.pi / 180, bbox3d[-3:])
    return bbox3d

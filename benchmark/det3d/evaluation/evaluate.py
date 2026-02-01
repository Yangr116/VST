import json
import os
from tqdm import tqdm
from loguru import logger
from pathlib import Path
import datetime
from collections import defaultdict
import re
import argparse
from benchmark.det3d.evaluation.dataset import COCO3D
from benchmark.det3d.evaluation.evaluator_3dod import _evaluate_predictions_on_omni, _derive_omni_results
from benchmark.det3d.evaluation.box_utils import compute_box3d_from_answer, compute_box2d_from_answer, create_intrinsics_from_dict


DEFAULT_FILTER_SETTINGS = {
    'truncation_thres': 1.0, 
    'visibility_thres': 0.0,
    'min_height_thres': 0.0,
    'modal_2D_boxes': False,
    'trunc_2D_boxes': True,
    'max_depth': 100000000.0,
    'max_height_thres': 10.5
}


def create_dummy_prediction(gtfile, savefile='./temp.json'):
    """Create a dummy prediction file for testing."""
    with open(gtfile, 'r') as f:
        data = json.load(f)

    results = [{'uid': item['uid'], 'answer': [item['answer'][0]]} for item in data]
    with open(savefile, 'w') as f:
        json.dump(results, f)
    print('A dummy prediction file has been saved to temp.json')


def create_coco_from_qa(gtfile_qa):
    """Create COCO format ground truth file from question-answer format."""
    with open(gtfile_qa, 'r', encoding='utf-8') as f:
        data = json.load(f)

    images_info = []
    annotations_info = []
    category_dict = defaultdict(lambda: len(category_dict) + 1)
    uid_to_image_id = {}
    uid_to_cat_id = {}
    uid_to_K = {}

    for data_item in tqdm(data):
        image_id = len(images_info) + 1
        image_info = {
            'uid': data_item['uid'],
            'image_path': data_item['image_path'],
            'id': image_id,
            'height': data_item['height'],
            'width': data_item['width'],
        }
        images_info.append(image_info)
        uid_to_image_id[data_item['uid']] = image_id
        uid_to_cat_id[data_item['uid']] = category_dict[data_item['category']]
        uid_to_K[data_item['uid']] = data_item['camera_intrinsics']

        for answer in data_item['answer']:
            ann_id = len(annotations_info) + 1
            bbox3d = compute_box3d_from_answer(answer)
            bbox2d_proj = compute_box2d_from_answer(answer, data_item['camera_intrinsics'])
            ann_info = {
                'id': ann_id,
                'image_id': image_id,
                'bbox2D_proj': bbox2d_proj.tolist(),
                'bbox3D_cam': bbox3d.tolist(),
                'centroid_cam': answer[:3],
                'dimensions': answer[3:6],
                'category_id': category_dict[data_item['category']],
                'answer': answer
            }
            annotations_info.append(ann_info)

    categories_info = [{'id': val, 'name': key} for key, val in category_dict.items()]
    results = {
        'images': images_info,
        'annotations': annotations_info,
        'categories': categories_info,
        'info': {'source': gtfile_qa}
    }
    with open('./temp_coco.json', 'w') as f:
        json.dump(results, f)
    print("A COCO format ground truth file has been saved to temp_coco.json")
    return uid_to_image_id, uid_to_K, uid_to_cat_id


def extract_bbox(input_string):
    results = []
    matches = re.findall(r'<3dbbox>(.*?)</3dbbox>', input_string)
    for _match in matches:
        _match = _match.split(' ')
        _match = [float(m) for m in _match]
        if len(_match) != 9:
            continue
        results.append(_match)
    return results


def load_gt_data(gtfile, filter_settings):
    """Load ground truth data and return necessary mappings."""
    if 'coco' in gtfile:
        coco3d_api = COCO3D([gtfile], filter_settings=filter_settings)
        uid_to_image_id, uid_to_K, uid_to_cat_id = None, None, None
        # TODO: 支持直接从 coco format gt file 中加载，适应 单个 question 有多个类别的场景
    else:
        uid_to_image_id, uid_to_K, uid_to_cat_id = create_coco_from_qa(gtfile)
        coco3d_api = COCO3D(['./temp_coco.json'], filter_settings=filter_settings)
    return coco3d_api, uid_to_image_id, uid_to_K, uid_to_cat_id


def prepare_instances(pred_file, uid_to_K, uid_to_image_id, uid_to_cat_id):
    logger.info(f"Preparing prediction from {pred_file}")
    with open(pred_file, 'r') as f:
        data = json.load(f)

    results = []
    for data_item in tqdm(data):
        if 'question_id' in data_item:
            qs_uid = data_item['question_id']  
        elif "id" in data_item:
            qs_uid = data_item['id']  
        else: 
            qs_uid = data_item['uid']
        K_dict = uid_to_K.get(qs_uid, None)
        if K_dict is None:
            logger.warning("This prediction doesn't have gt, skip")
            continue
        K = create_intrinsics_from_dict(K_dict)
        answers = data_item.get('predictions', data_item.get('answer', None))
        if isinstance(answers, str):
            # extra bbox from string
            answers = extract_bbox(answers)
        for answer in answers:
            bbox3d = compute_box3d_from_answer(answer)
            bbox2d = compute_box2d_from_answer(answer, K)
            result = {
                'image_id': uid_to_image_id[qs_uid],
                'category_id': uid_to_cat_id[qs_uid],
                'bbox': bbox2d.tolist(),
                'bbox3D': bbox3d.tolist(),
                'score': 1.0,
                'depth': answer[2],
                'prediction': answer
            }
            results.append(result)
    return results


def evaluate(
        gtfile: str, 
        pred_file: str = './temp.json', 
        outdir: str = './work_dirs/evaluate', 
        only_3d: bool = True,
        filter_settings: dict = DEFAULT_FILTER_SETTINGS):
    """
    Evaluate predictions against ground truth data.
    Args:
        gtfile: the ground truth file saving the data, it supports coco format file and question format file
        pred_file: the results file from the lmm, we will parse the results from it to coco instance format
                        e.g.
                        result = {
                            'image_id': xxx,
                            'category_id': xxx,
                            'bbox': xxx,  # list, len=4, in the pixel coordinate
                            'bbox3D': xxx, # list, len=8, in the camera coordinate
                            'score': 1.0,
                            'depth': xxx,  # in the camera coordinate
                        }
        outdir: the path to save the evaluation results
        filter_settings: some settings from omni3d
    """
    os.makedirs(outdir, exist_ok=True)
    time_flag = datetime.datetime.now().strftime("%Y%m%d%H%M")
    log_file = os.path.join(outdir,  time_flag + '-' + Path(pred_file).stem + '.log')
    logger.add(log_file)

    ### Load gt file ###
    logger.info(f"Load gt data from {gtfile}")
    coco3d_api, uid_to_image_id, uid_to_K, uid_to_cat_id = load_gt_data(gtfile, filter_settings)
    category_name_to_id = {val['name']: key for key, val in coco3d_api.cats.items()}

    ### prepare the prediction instances from prediction file ###
    results = prepare_instances(pred_file, uid_to_K, uid_to_image_id, uid_to_cat_id)

    logger.info(f"\nPrediction: {results[0]}\nGT:{coco3d_api.imgToAnns[results[0]['image_id']]}")
    
    # start to evaluate
    iou_type = 'bbox'
    task = 'bbox'

    evals, log_strs = _evaluate_predictions_on_omni(
        coco3d_api,
        results,
        img_ids=None,
        iou_type=iou_type,
        eval_prox=False,
        only_3d=only_3d,
    )

    class_names = list(category_name_to_id.keys())
    for mode in evals.keys():
        _derive_omni_results(
            evals[mode],
            task,
            mode,
            class_names=class_names,
        )

    if '2D' in log_strs:
        logger.info('\n' + log_strs['2D'])

    if '3D' in log_strs:
        logger.info('\n' + log_strs['3D'])

    return None


if __name__ == '__main__':
    filter_settings = {
        'truncation_thres': 1.0,  # 如果截断高于 truncation_thres  就 ignore
        'visibility_thres': 0.0,
        'min_height_thres': 0.0,
        'modal_2D_boxes': False,
        'trunc_2D_boxes': True,
        'max_depth': 100000000.0,
        'max_height_thres': 10.5
    }

    parser = argparse.ArgumentParser(description="evaluation")
    parser.add_argument('--gt', type=str, required=True)
    parser.add_argument('--pred', type=str, required=True)
    parser.add_argument('--outdir', type=str, default='work_dirs/evaluate')
    args = parser.parse_args()
    evaluate(
        gtfile=args.gt, 
        pred_file=args.pred, 
        outdir=args.outdir,
        filter_settings=filter_settings)

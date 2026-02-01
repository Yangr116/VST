"""
This dataset is only used for evaluation.
Adapted from omni3D
"""

import time
import json
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO
from collections import defaultdict
from benchmark.det3d.evaluation.box_utils import boxconverter_xyxy2xywh


def is_ignore(anno, filter_settings, image_height):

    ignore = anno.get('behind_camera', False)
    ignore |= (not bool(anno.get('valid3D', -1)))

    if ignore:
        return ignore

    ignore |= anno['dimensions'][0] <= 0
    ignore |= anno['dimensions'][1] <= 0
    ignore |= anno['dimensions'][2] <= 0
    ignore |= anno.get('center_cam', anno.get('centroid_cam'))[2] > filter_settings['max_depth']
    # ignore |= (anno['lidar_pts'] == 0)
    # ignore |= (anno['segmentation_pts'] == 0)
    # ignore |= (anno['depth_error'] > 0.5)
    
    # tightly annotated 2D boxes are not always available.
    if filter_settings['modal_2D_boxes'] and 'bbox2D_tight' in anno and anno['bbox2D_tight'][0] != -1:
        bbox2D =  boxconverter_xyxy2xywh(anno['bbox2D_tight'])

    # truncated projected 2D boxes are also not always available.
    elif filter_settings['trunc_2D_boxes'] and 'bbox2D_trunc' in anno and not np.all([val==-1 for val in anno['bbox2D_trunc']]):
        bbox2D =  boxconverter_xyxy2xywh(anno['bbox2D_trunc'])

    # use the projected 3D --> 2D box, which requires a visible 3D cuboid.
    elif 'bbox2D_proj' in anno:
        bbox2D =  boxconverter_xyxy2xywh(anno['bbox2D_proj'])

    else:
        bbox2D = anno['bbox']

    # 过滤掉了边缘的 obj
    ignore |= bbox2D[3] <= filter_settings['min_height_thres']*image_height
    ignore |= bbox2D[3] >= filter_settings['max_height_thres']*image_height

    ignore |= (anno.get('truncation', 0.0) >=0 and anno.get('truncation', 0.0) >= filter_settings['truncation_thres'])
    ignore |= (anno.get('visibility', 1.0) >= 0 and anno.get('visibility', 1.0) <= filter_settings['visibility_thres'])
    
    if 'ignore_names' in filter_settings:
        ignore |= anno.get('category_name', anno.get('category')) in filter_settings['ignore_names']

    return ignore


class COCO3D(COCO):
    '''
    Class for COCO-like dataset object. Not inherently related to 
    use with Detectron2 or training per se. 
    '''

    def __init__(self, annotation_files, filter_settings=None):
             
        # load dataset
        self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)

        if isinstance(annotation_files, str):
            annotation_files = [annotation_files,]
        
        cats_ids_master = []
        cats_master = []
        
        for annotation_file in annotation_files:

            name = str(Path(annotation_file).stem)

            print('loading {} annotations into memory...'.format(name))
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time()- tic))

            if type(dataset['info']) == list:
                dataset['info'] = dataset['info'][0]
                
            dataset['info']['known_category_ids'] = [cat['id'] for cat in dataset['categories']]

            # first dataset
            if len(self.dataset) == 0:
                self.dataset = dataset
            
            # concatenate datasets
            else:

                if type(self.dataset['info']) == dict:
                    self.dataset['info'] = [self.dataset['info']]
                    
                self.dataset['info'] += [dataset['info']]
                self.dataset['annotations'] += dataset['annotations']
                self.dataset['images'] += dataset['images']
            
            # sort through categories
            for cat in dataset['categories']:

                if not cat['id'] in cats_ids_master:
                    cats_ids_master.append(cat['id'])
                    cats_master.append(cat)

        if filter_settings is None:

            # include every category in the master list
            self.dataset['categories'] = [
                cats_master[i] 
                for i in np.argsort(cats_ids_master) 
            ]
        elif 'ignore_names' not in filter_settings or 'category_names' not in filter_settings:
            
            # include every category in the master list
            self.dataset['categories'] = [
                cats_master[i] 
                for i in np.argsort(cats_ids_master) 
            ]

            # we need to prepare the bbox2d and bbox3d
            valid_anns = []
            im_height_map = {}

            for im_obj in self.dataset['images']:
                im_height_map[im_obj['id']] = im_obj['height']

            # Filter out annotations
            for anno_idx, anno in enumerate(self.dataset['annotations']):
                
                im_height = im_height_map[anno['image_id']]
                
                ignore = is_ignore(anno, filter_settings, im_height)
                # ignore = False
                
                if filter_settings['trunc_2D_boxes'] and 'bbox2D_trunc' in anno and not np.all([val==-1 for val in anno['bbox2D_trunc']]):
                    bbox2D =  boxconverter_xyxy2xywh(anno['bbox2D_trunc'])

                elif anno.get('bbox2D_proj', None) is not None and anno['bbox2D_proj'][0] != -1:
                    bbox2D = boxconverter_xyxy2xywh(anno['bbox2D_proj'])

                elif anno.get('bbox2D_tight', None) is not None and anno['bbox2D_tight'][0] != -1:
                    bbox2D = boxconverter_xyxy2xywh(anno['bbox2D_tight'])

                else: 
                    continue

                width = bbox2D[2]
                height = bbox2D[3]

                self.dataset['annotations'][anno_idx]['area'] = width*height
                self.dataset['annotations'][anno_idx]['iscrowd'] = False
                self.dataset['annotations'][anno_idx]['ignore'] = ignore
                self.dataset['annotations'][anno_idx]['ignore2D'] = ignore
                self.dataset['annotations'][anno_idx]['ignore3D'] = ignore
                
                if filter_settings['modal_2D_boxes'] and anno['bbox2D_tight'][0] != -1:
                    self.dataset['annotations'][anno_idx]['bbox'] = boxconverter_xyxy2xywh(anno['bbox2D_tight'])
                
                else:
                    self.dataset['annotations'][anno_idx]['bbox'] = bbox2D
                
                self.dataset['annotations'][anno_idx]['bbox3D'] = anno['bbox3D_cam']
                self.dataset['annotations'][anno_idx]['depth'] = anno.get('center_cam', anno.get('centroid_cam'))[2]
                category_name = anno.get('category_name', anno.get('category',))
                valid_anns.append(self.dataset['annotations'][anno_idx])

            self.dataset['annotations'] = valid_anns

        else:
        
            # determine which categories we may actually use for filtering.
            trainable_cats = set(filter_settings['ignore_names']) | set(filter_settings['category_names'])

            # category names are provided to us
            if len(filter_settings['category_names']) > 0:

                self.dataset['categories'] = [
                    cats_master[i] 
                    for i in np.argsort(cats_ids_master) 
                    if cats_master[i]['name'] in filter_settings['category_names']
                ]

            # no categories are provided, so assume use ALL available.
            else:

                self.dataset['categories'] = [
                    cats_master[i] 
                    for i in np.argsort(cats_ids_master) 
                ]

                filter_settings['category_names'] = [cat['name'] for cat in self.dataset['categories']]

                trainable_cats = trainable_cats | set(filter_settings['category_names'])
            
            valid_anns = []
            im_height_map = {}

            for im_obj in self.dataset['images']:
                im_height_map[im_obj['id']] = im_obj['height']

            # Filter out annotations
            for anno_idx, anno in enumerate(self.dataset['annotations']):
                
                im_height = im_height_map[anno['image_id']]
                
                ignore = is_ignore(anno, filter_settings, im_height)
                
                if filter_settings['trunc_2D_boxes'] and 'bbox2D_trunc' in anno and not np.all([val==-1 for val in anno['bbox2D_trunc']]):
                    bbox2D =  boxconverter_xyxy2xywh(anno['bbox2D_trunc'])

                elif anno['bbox2D_proj'][0] != -1:
                    bbox2D = boxconverter_xyxy2xywh(anno['bbox2D_proj'])

                elif anno['bbox2D_tight'][0] != -1:
                    bbox2D = boxconverter_xyxy2xywh(anno['bbox2D_tight'])

                else: 
                    continue

                width = bbox2D[2]
                height = bbox2D[3]

                self.dataset['annotations'][anno_idx]['area'] = width*height
                self.dataset['annotations'][anno_idx]['iscrowd'] = False
                self.dataset['annotations'][anno_idx]['ignore'] = ignore
                self.dataset['annotations'][anno_idx]['ignore2D'] = ignore
                self.dataset['annotations'][anno_idx]['ignore3D'] = ignore
                
                if filter_settings['modal_2D_boxes'] and anno['bbox2D_tight'][0] != -1:
                    self.dataset['annotations'][anno_idx]['bbox'] = boxconverter_xyxy2xywh(anno['bbox2D_tight'])
                
                else:
                    self.dataset['annotations'][anno_idx]['bbox'] = bbox2D
                
                self.dataset['annotations'][anno_idx]['bbox3D'] = anno['bbox3D_cam']
                self.dataset['annotations'][anno_idx]['depth'] = anno.get('center_cam', anno.get('centroid_cam'))[2]

                category_name = anno.get('category_name', anno.get('category',))

                # category is part of trainable categories?
                if category_name in trainable_cats:
                    valid_anns.append(self.dataset['annotations'][anno_idx])

            self.dataset['annotations'] = valid_anns

        self.createIndex()

    def info(self):
        
        infos = self.dataset['info']
        if type(infos) == dict:
            infos = [infos]

        for i, info in enumerate(infos):
            print('Dataset {}/{}'.format(i+1, infos))

            for key, value in info.items():
                print('{}: {}'.format(key, value))

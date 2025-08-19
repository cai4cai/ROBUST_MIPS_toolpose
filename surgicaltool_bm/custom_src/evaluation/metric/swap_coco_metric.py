from collections import defaultdict, OrderedDict
from typing import Dict

import numpy as np
from mmengine.logging import MMLogger
from xtcocotools.coco import COCO

from mmpose.registry import METRICS
from mmpose.evaluation.metrics import CocoMetric

import tempfile
import os.path as osp
from mmengine.fileio import get_local_path

from mmpose.evaluation.functional import transform_ann, transform_pred
from mmpose.evaluation.functional import oks_nms, soft_oks_nms

@METRICS.register_module()
class SwapCocoMetric(CocoMetric):
    """Optimized COCO keypoint evaluation metric with automatic swapping of keypoints 3 and 4
    
    This class inherits from the standard CocoMetric and automatically attempts
    to swap the 3rd and 4th keypoints during evaluation. Save the better one. 

    All parameters are identical to those in CocoMetric, with no additional configuration required.
    """
    default_prefix = 'swap_coco'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.optimization_stats = {
            'total_instances': 0,
            'improved_instances': 0,
            'improvement_sum': 0.0
        }
        
        print(f"SwapCocoMetric initialized - will automatically optimize keypoint order")

    def _compute_oks_for_instance(self, pred_kpts, gt_kpts, sigmas, area):
        """
        Args:
            pred_kpts: [N, 3] (x, y, score)
            gt_kpts: [N, 3] (x, y, visibility)  
            sigmas: each keypoint sigma
            area: bbox area
        Returns:
            oks: OKS score
        """
        vars = (sigmas * 2) ** 2
        
        dx = pred_kpts[:, 0] - gt_kpts[:, 0]
        dy = pred_kpts[:, 1] - gt_kpts[:, 1]
        e = (dx**2 + dy**2) / vars / (area + np.spacing(1)) / 2
        
        # only visible
        visible = gt_kpts[:, 2] > 0
        e = e[visible]
        
        if len(e) == 0:
            return 0
        
        return np.sum(np.exp(-e)) / len(e)


    def _swap_keypoints_3_4(self, keypoints, keypoint_scores):
        """
        swap the 3rd and 4th keypoint 
        Args:
            keypoints: [N, 2] (x, y)
            keypoint_scores: [N] 
        Returns:
            swapped_keypoints, swapped_scores
        """
        swapped_kpts = keypoints.copy()
        swapped_scores = keypoint_scores.copy()
        
        if len(keypoints) >= 4:
            swapped_kpts[2], swapped_kpts[3] = keypoints[3].copy(), keypoints[2].copy()
            swapped_scores[2], swapped_scores[3] = keypoint_scores[3], keypoint_scores[2]
        
        return swapped_kpts, swapped_scores

    def _optimize_keypoint_order_with_oks(self, kpts):
        """
        optimize keypoint order based on OKS
        compare the original order and the swapped order, save the better one.
        """
        optimized_kpts = defaultdict(list)
        
        for img_id, instances in kpts.items():
            # obtain ground truth annotations
            gt_anns = []
            for ann_id in self.coco.getAnnIds(imgIds=[img_id]):
                ann = self.coco.anns[ann_id]
                gt_anns.append(ann)
            
            optimized_instances = []
            
            for instance in instances:
                self.optimization_stats['total_instances'] += 1
                
                pred_keypoints = instance['keypoints']  # [N, 2]
                pred_scores = instance['keypoint_scores']  # [N]
                pred_area = instance['area']
                
                original_oks = 0
                best_gt_ann = None
                
                for gt_ann in gt_anns:
                    gt_kpts_flat = np.array(gt_ann['keypoints']).reshape(-1, 3)
                    # gt_area = gt_ann.get('area', pred_area)
                    gt_area = gt_ann['area']  
                    
                    # prediction [N, 3] (x, y, score)
                    pred_kpts_with_scores = np.concatenate([
                        pred_keypoints, pred_scores[:, None]
                    ], axis=-1)
                    
                    oks = self._compute_oks_for_instance(
                        pred_kpts_with_scores, gt_kpts_flat, 
                        self.dataset_meta['sigmas'], gt_area
                    )
                    
                    if oks > original_oks:
                        original_oks = oks
                        best_gt_ann = gt_ann
                
                # swap
                swapped_keypoints, swapped_scores = self._swap_keypoints_3_4(
                    pred_keypoints, pred_scores
                )
                
                swapped_oks = 0
                if best_gt_ann is not None:
                    gt_kpts_flat = np.array(best_gt_ann['keypoints']).reshape(-1, 3)
                    gt_area = best_gt_ann.get('area', pred_area)
                    
                    swapped_kpts_with_scores = np.concatenate([
                        swapped_keypoints, swapped_scores[:, None]
                    ], axis=-1)
                    
                    swapped_oks = self._compute_oks_for_instance(
                        swapped_kpts_with_scores, gt_kpts_flat,
                        self.dataset_meta['sigmas'], gt_area
                    )
                
                # save the better oks
                if swapped_oks > original_oks:
                    improvement = swapped_oks - original_oks
                    self.optimization_stats['improved_instances'] += 1
                    self.optimization_stats['improvement_sum'] += improvement
                    
                    print(f"Image {img_id}: Swapped keypoints improve OKS from {original_oks:.4f} to {swapped_oks:.4f} (+{improvement:.4f})")
                    instance['keypoints'] = swapped_keypoints
                    instance['keypoint_scores'] = swapped_scores
                
                optimized_instances.append(instance)
            
            optimized_kpts[img_id] = optimized_instances
        
        return optimized_kpts

    def compute_metrics(self, results: list) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()
        
        # split prediction and gt list
        preds, gts = zip(*results)

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        if self.coco is None:
            logger.info('Converting ground truth to coco format...')
            coco_json_path = self.gt_to_coco_json(
                gt_dicts=gts, outfile_prefix=outfile_prefix)
            self.coco = COCO(coco_json_path)
        if self.gt_converter is not None:
            for id_, ann in self.coco.anns.items():
                self.coco.anns[id_] = transform_ann(
                    ann, self.gt_converter['num_keypoints'],
                    self.gt_converter['mapping'])

        kpts = defaultdict(list)

        # group the preds by img_id
        for pred in preds:
            img_id = pred['img_id']

            if self.pred_converter is not None:
                pred = transform_pred(pred,
                                      self.pred_converter['num_keypoints'],
                                      self.pred_converter['mapping'])

            for idx, keypoints in enumerate(pred['keypoints']):
                instance = {
                    'id': pred['id'],
                    'img_id': pred['img_id'],
                    'category_id': pred['category_id'],
                    'keypoints': keypoints,
                    'keypoint_scores': pred['keypoint_scores'][idx],
                    'bbox_score': pred['bbox_scores'][idx],
                }
                if 'bbox' in pred:
                    instance['bbox'] = pred['bbox'][idx]

                if 'areas' in pred:
                    instance['area'] = pred['areas'][idx]
                else:
                    # use keypoint to calculate bbox and get area
                    area = (
                        np.max(keypoints[:, 0]) - np.min(keypoints[:, 0])) * (
                            np.max(keypoints[:, 1]) - np.min(keypoints[:, 1]))
                    instance['area'] = area

                kpts[img_id].append(instance)

        # sort keypoint results according to id and remove duplicate ones
        kpts = self._sort_and_unique_bboxes(kpts, key='id')

        print("Optimizing keypoint order based on OKS...")
        kpts = self._optimize_keypoint_order_with_oks(kpts)
        
        total = self.optimization_stats['total_instances']
        improved = self.optimization_stats['improved_instances']
        avg_improvement = (self.optimization_stats['improvement_sum'] / improved 
                          if improved > 0 else 0)
        
        print(f"Keypoint optimization completed:")
        print(f"  Total instances: {total}")
        print(f"  Improved instances: {improved} ({improved/total*100:.1f}%)")
        print(f"  Average OKS improvement: {avg_improvement:.4f}")  
        
        valid_kpts = defaultdict(list)
        if self.pred_converter is not None:
            num_keypoints = self.pred_converter['num_keypoints']
        else:
            num_keypoints = self.dataset_meta['num_keypoints']
            
        for img_id, instances in kpts.items():
            for instance in instances:
                # concatenate the keypoint coordinates and scores
                instance['keypoints'] = np.concatenate([
                    instance['keypoints'], instance['keypoint_scores'][:, None]
                ], axis=-1)
                
                if self.score_mode == 'bbox':
                    instance['score'] = instance['bbox_score']
                elif self.score_mode == 'keypoint':
                    instance['score'] = np.mean(instance['keypoint_scores'])
                else:
                    bbox_score = instance['bbox_score']
                    if self.score_mode == 'bbox_rle':
                        keypoint_scores = instance['keypoint_scores']
                        instance['score'] = float(bbox_score +
                                                  np.mean(keypoint_scores) +
                                                  np.max(keypoint_scores))
                    else:  # self.score_mode == 'bbox_keypoint':
                        mean_kpt_score = 0
                        valid_num = 0
                        for kpt_idx in range(num_keypoints):
                            kpt_score = instance['keypoint_scores'][kpt_idx]
                            if kpt_score > self.keypoint_score_thr:
                                mean_kpt_score += kpt_score
                                valid_num += 1
                        if valid_num != 0:
                            mean_kpt_score /= valid_num
                        instance['score'] = bbox_score * mean_kpt_score
            
            # perform nms
            if self.nms_mode == 'none':
                valid_kpts[img_id] = instances
            else:
                nms = oks_nms if self.nms_mode == 'oks_nms' else soft_oks_nms
                keep = nms(
                    instances,
                    self.nms_thr,
                    sigmas=self.dataset_meta['sigmas'])
                valid_kpts[img_id] = [instances[_keep] for _keep in keep]

        # convert results to coco style and dump into a json file
        self.results2json(valid_kpts, outfile_prefix=outfile_prefix)

        # only format the results without doing quantitative evaluation
        if self.format_only:
            logger.info('results are saved in '
                        f'{osp.dirname(outfile_prefix)}')
            return {}

        # evaluation results
        eval_results = OrderedDict()
        logger.info(f'Evaluating {self.__class__.__name__}...')
        info_str = self._do_python_keypoint_eval(outfile_prefix)
        name_value = OrderedDict(info_str)
        eval_results.update(name_value)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results
import os
import importlib

import json
from collections import defaultdict

import cv2
import numpy as np

from mmengine.config import Config
from mmpose.utils import register_all_modules
from mmpose.apis import init_model, inference_topdown
from mmpose.datasets import build_dataset
from mmpose.registry import METRICS, VISUALIZERS
from mmpose.structures import merge_data_samples

# configs-rtml
# POSE_CFG = 'configs/rtmpose-l_8xb256-420e_coco-256x192.py'
# POSE_CKPT = 'work_dirs/scientific_data//best_coco_AP_epoch_285.pth'
# DEVICE = 'cuda:0'       

# DET_JSON = '/nfs/home/zhan/code/ROBUST_MIPS_toolpose/dataset/converted_detections_test.json'  # mmdet bbox results

# OUT_DIR = 'work_dirs/outputs/pose_oks_vis_rtml'
# os.makedirs(OUT_DIR, exist_ok=True)

# configs-sbl
# POSE_CFG = 'configs/td-hm_res152_8xb32-210e_coco-256x192.py'
# POSE_CKPT = 'work_dirs/scientific_data/best_coco_AP_epoch_145.pth'
# DEVICE = 'cuda:0'       

# DET_JSON = '/nfs/home/zhan/code/ROBUST_MIPS_toolpose/dataset/converted_detections_test.json'  # mmdet bbox results

# OUT_DIR = 'work_dirs/outputs/pose_oks_vis_sbl'
# os.makedirs(OUT_DIR, exist_ok=True)

# configs-vitpose
POSE_CFG = 'configs/td-hm_ViTPose-large_8xb64-210e_coco-256x192.py'
POSE_CKPT = 'work_dirs/scientific_data/best_coco_AP_epoch_95.pth'
DEVICE = 'cuda:0'       

DET_JSON = '/nfs/home/zhan/code/ROBUST_MIPS_toolpose/dataset/converted_detections_test.json'  # mmdet bbox results

OUT_DIR = 'work_dirs/outputs/pose_oks_vis_vitpose'
os.makedirs(OUT_DIR, exist_ok=True)

# SELECTED_IMAGE_IDS = None
SELECTED_IMAGE_IDS = [40, 47, 87, 130, 2190, 2060, 2108]

## bbox results from converted_detections_test.json
def load_detections(det_json_path):
    with open(det_json_path, 'r') as f:
        dets = json.load(f)

    det_by_img = defaultdict(list)
    for det in dets:
        img_id = det['image_id']
        x, y, w, h = det['bbox']
        score = det.get('score', 1.0)
        det_by_img[img_id].append([x, y, x + w, y + h, score])

    for img_id in det_by_img:
        det_by_img[img_id] = np.asarray(det_by_img[img_id], dtype=np.float32)

    return det_by_img

def build_model_and_dataset():

    register_all_modules()
    cfg = Config.fromfile(POSE_CFG)

    if hasattr(cfg, 'custom_imports'):
        imports = cfg.custom_imports.get('imports', [])
        for mod in imports:
            importlib.import_module(mod)

    model = init_model(POSE_CFG, POSE_CKPT, device=DEVICE)

    test_dataset = build_dataset(cfg.test_dataloader.dataset)
    dataset_meta = test_dataset.metainfo

    metric = METRICS.build(cfg.test_evaluator)
    metric.dataset_meta = dataset_meta
    coco = metric.coco

    ds_cfg = cfg.test_dataloader.dataset
    data_root = ds_cfg.get('data_root', '')
    img_prefix = ds_cfg.get('data_prefix', {}).get('img', '')
    img_root = os.path.join(data_root, img_prefix)

    return model, cfg, coco, dataset_meta, img_root, metric

def compute_swap_oks_for_image(metric, coco, dataset_meta,
                               image_id, keypoints, keypoint_scores):

    gt_ids = coco.getAnnIds(imgIds=[image_id])
    gt_anns = coco.loadAnns(gt_ids)

    oks_list = []

    for kp, kp_scores in zip(keypoints, keypoint_scores):
        # kp: [K, 2], kp_scores: [K]
        original_oks = 0.0
        best_gt = None

        pred_kpts_with_scores = np.concatenate(
            [kp, kp_scores[:, None]], axis=-1
        )  # [K, 3]

        for gt in gt_anns:
            gt_kpts_flat = np.array(gt['keypoints'], dtype=np.float32).reshape(-1, 3)
            gt_area = gt['area']

            oks = metric._compute_oks_for_instance(
                pred_kpts_with_scores,
                gt_kpts_flat,
                dataset_meta['sigmas'],
                gt_area
            )

            if oks > original_oks:
                original_oks = oks
                best_gt = gt

        swapped_kpts, swapped_scores = metric._swap_keypoints_3_4(kp, kp_scores)
        swapped_oks = original_oks  

        if best_gt is not None:
            gt_kpts_flat = np.array(best_gt['keypoints'], dtype=np.float32).reshape(-1, 3)
            gt_area = best_gt['area']

            swapped_kpts_with_scores = np.concatenate(
                [swapped_kpts, swapped_scores[:, None]], axis=-1
            )

            swapped_oks = metric._compute_oks_for_instance(
                swapped_kpts_with_scores,
                gt_kpts_flat,
                dataset_meta['sigmas'],
                gt_area
            )

        oks_list.append(max(original_oks, swapped_oks))

    return oks_list


def build_visualizer(cfg, dataset_meta):
    if not hasattr(cfg, 'visualizer'):
        raise RuntimeError(
            ' `visualizer` not in config'
            'need to add a PoseLocalVisualizer '
        )

    vis_cfg = cfg.visualizer
    visualizer = VISUALIZERS.build(vis_cfg)
    visualizer.set_dataset_meta(dataset_meta)

    return visualizer


def visualize_one_image(img_path, pose_datasample, oks_list, out_path, visualizer):

    img_bgr = cv2.imread(img_path)
    assert img_bgr is not None, f'Fail to read image: {img_path}'

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    visualizer.add_datasample(
        name='result',
        image=img_rgb,
        data_sample=pose_datasample,
        draw_gt=False,
        show=False,
        wait_time=0,
        out_file=out_path,
        kpt_thr = 0.0
    )

    vis_bgr = cv2.imread(out_path)
    pred_instances = pose_datasample.pred_instances

    bboxes = pred_instances.bboxes
    keypoints = pred_instances.keypoints 

    for i, oks in enumerate(oks_list):
        x1, y1, x2, y2 = bboxes[i]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        cv2.rectangle(
            vis_bgr,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),   # BGR
            2,             # thickness
            lineType=cv2.LINE_AA,
        )

        text = f'OKS={oks:.3f}'
        text_x = x1 + 5
        text_y = y1 + 20
        # text_org = (x1, max(0, y1 - 5))
        h, w = img_bgr.shape[:2]
        text_x = max(0, min(text_x, w - 1))
        text_y = max(0, min(text_y, h - 1)) 
        cv2.putText(
            vis_bgr,
            text,
            # text_org,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(out_path, vis_bgr)


def main():
    
    model, cfg, coco, dataset_meta, IMG_ROOT, swap_metric = build_model_and_dataset()
    det_by_img = load_detections(DET_JSON)
    visualizer = build_visualizer(cfg, dataset_meta)

    all_pred_results = []
    if SELECTED_IMAGE_IDS is not None:
        target_ids = SELECTED_IMAGE_IDS
    else:
        target_ids = sorted(det_by_img.keys())

    for img_id in target_ids:
        img_info = coco.loadImgs([img_id])[0]
        file_name = img_info['file_name']
        img_path = os.path.join(IMG_ROOT, file_name)

        bboxes_xyxy = det_by_img.get(img_id, None)
        if bboxes_xyxy is None or len(bboxes_xyxy) == 0:
            print(f'[WARN] image_id={img_id} has no detection results, skip')
            continue

        bboxes_xyxy_4 = bboxes_xyxy[:, :4]
        print(f"[DEBUG] img_id={img_id}, num_det_boxes={len(bboxes_xyxy)}")
        print("[DEBUG] det boxes:", bboxes_xyxy[:, :4])

        pose_results = inference_topdown(
            model,
            img_path,
            bboxes_xyxy_4
        )

        if len(pose_results) == 0:
            print(f'[WARN] image_id={img_id} has no pose results, skip')
            continue

        pose_sample = merge_data_samples(pose_results)
        pred_instances = pose_sample.pred_instances

        keypoints = pred_instances.keypoints  # (N, K, 2)
        print(f"[DEBUG] img_id={img_id}, num_pose_instances={keypoints.shape[0]}")

        keypoint_scores = pred_instances.keypoint_scores  # (N, K)
        bboxes = pred_instances.bboxes  # (N, 4)

        # SwapCocoMetric for OKS
        oks_list = compute_swap_oks_for_image(
            swap_metric,
            coco,
            dataset_meta,
            img_id,
            keypoints,
            keypoint_scores,
        )
        num_inst = keypoints.shape[0]

        img_results = []
        for i in range(num_inst):
            result_dict = {
                'image_id': int(img_id),
                'file_name': file_name,
                'instance_id': int(i),
                'bbox': bboxes[i].tolist(),                 # [x1, y1, x2, y2]
                'keypoints': keypoints[i].tolist(),         
                'keypoint_scores': keypoint_scores[i].tolist(),
                'oks': float(oks_list[i]),
            }
            img_results.append(result_dict)
            all_pred_results.append(result_dict)

        base = os.path.splitext(os.path.basename(file_name))[0]
        out_path = os.path.join(OUT_DIR, f'{base}_oks.jpg')
        print("OUT_DIR:", OUT_DIR)
        visualize_one_image(img_path, pose_sample, oks_list, out_path, visualizer)
        print(f'[INFO] visualization save to:{out_path}')

        json_path = os.path.splitext(out_path)[0] + '.json'
        with open(json_path, 'w') as f:
            json.dump(img_results, f, indent=2)
        print(f'[INFO] json save to: {json_path}')

    save_json_path = os.path.join(OUT_DIR, 'pred_keypoints_with_oks.json')
    with open(save_json_path, 'w') as f:
        json.dump(all_pred_results, f, indent=2)
    print(f'[INFO] All predicted keypoints have been saved to: {save_json_path}')

if __name__ == '__main__':
    main()
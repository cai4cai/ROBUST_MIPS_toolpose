# ROBUST-MIPS: A Combined Skeletal Pose and Instance Segmentation Dataset for Laparoscopic Surgical Instruments

This repository provides implementation of surgical-tool pose estimation (based on **ROBUST-MIPS**).
Built on top of [MMPose v1.3.0](https://github.com/open-mmlab/mmpose).

## Introduction
Localisation of surgical tools constitutes a foundational building block for computer-assisted interventional technologies. Works in this field typically focus on training deep learning models to perform segmentation tasks. Performance of learning-based approaches is limited by the availability of diverse annotated data. We argue that skeletal pose annotations are a more efficient annotation approach for surgical tools, striking a balance between richness of semantic information and ease of annotation, thus allowing for accelerated growth of available annotated data. To encourage adoption of this annotation style, we present, ROBUST-MIPS, a combined tool pose and tool instance segmentation dataset derived from the existing ROBUST-MIS dataset. Our enriched dataset facilitates the joint study of these two annotation styles and allow head-to-head comparison on various downstream tasks.
To demonstrate the adequacy of pose annotations for surgical tool localisation, we set up a simple benchmark using popular pose estimation methods and observe high-quality results. To ease adoption, together with the dataset, we release our benchmark models and custom tool pose annotation software.

[🔗 Download ROBUST_MIPS_toolpose](https://github.com/cai4cai/ROBUST_MIPS_toolpose)

[🔗 Download tool-pose-annotation-gui](https://github.com/cai4cai/tool-pose-annotation-gui)

[🔗 Download ROBUST_MIPS Dataset](https://doi.org/10.7303/syn64023381)


## Model Configuration

The configuration files for the benchmarked models are located in:  
`ROBUST_MIPS_toolpose/surgicaltool_bm/configs`
Since this project is built upon MMPose (which defaults to human pose estimation), specific overrides are required to adapt it for the ROBUST-MIPS dataset.

### dataset_info

Taking the ROBUST-MIPS dataset as an example. `keypoint_info` defines the keypoints, `skeleton_info` defines the connection relationships between keypoints, and `sigmas` represents the per-keypoint localization uncertainty used for OKS calculation.

### number fo keypoints

set `NUM_KEYPOINTS=4`, and update it in the `model` configuration

### bbox_file

Since the Top-Down paradigm is a two-stage approach, the first stage requires generating bounding box detection results. In this implementation, mmdetection was used to porvided the bbox detection results. The corresponding results should be updated in the `val_dataloader` and `test_dataloader`. 

### test_evaluator

Given the unordered nature of the surgical tool endpoints, when evaluating prediction results, we should select the optimal match for the tip pairs (considering both the standard and swapped assignments of Tip1 and Tip2).  This metric is implemented via `swap_coco_metric` (`surgicaltool_bm/custom_src/evaluation/metric/swap_coco_metric.py`), which needs to be imported in the config file using `custom_imports`.

## Customize Dataset
Since our annotation software (tool-pose-annotation-gui) generates a separate JSON annotation file for each image, we need to aggregate these independent, single-image annotation files into three unified annotation files corresponding to the training, validation, and testing sets. The aggregated format should align with COCO. 

Importantly, during the generation of these annotation files, we modified the formula for calculating `area`. Instead of storing the standard area, we store $\frac{\mathrm{width}^2 + \mathrm{height}^2}{2}$. This conversion process can be completed using the scripts provided in the `utilities` folder.

## Overview

This project extends the MMPose v1.3.0 framework to support:

- **Custom dataset**: `ROBUST-MIPS` for laparoscopic frames  
- **Custom evaluation metrics**: custom OKS metric parameters `swap_coco_metric`    
- **Plug-and-play interface**: uses MMPose’s `tools/train.py` & `tools/test.py`  

## Docker

We provide a `Dockerfile` to lock in your Python/CUDA environment and install all dependencies.


## Prerequisites

Before you begin, ensure you have met the following requirements:

- **Docker image** Build the image based on the Dockerfile
- **MMPose** v1.3.0 installed (fork your copy and install in editable mode)  


## Table of Contents 
- [Project Structure](#project-structure) 
- [ROBUST-MIPS Dataset](#data) 
- [License](#license) 

## Project Structure 
``` 
ROBUST_MIPS_toolpose/
├── surgicaltool_bm/        # Main package for surgical tool pose estimation
│   ├── configs/            # Configuration files
│   ├── custom_src/         # Custom source code and extensions
│   ├── tools/              # Utility scripts and training/testing tools
│   └── setup.py            # Package setup script
│
├── dataset/                # ROBUST-MIPS
├── mmpose/                 # MMPose lib
├── utilities/              # Processing tools for ROBUST-MIPS datasets
│
├── .dockerignore           # Files ignored by Docker build
├── .gitignore              # Files ignored by Git
├── build.sh                # Shell script to build the environment
├── Dockerfile              # Docker build file
└── README.md               # Project documentation (this file)

``` 

### utilities/reorganizedata.py

The `reorganizedata.py` script restructures the dataset, originally organized in a hierarchical format as `Training/Testing -> Surgery type -> Procedure -> Frame -> Images and annotations`. It consolidates images and JSON annotation files into a simplified format:

- `rename_training/img` and `rename_training/json` for training images and annotations
- `rename_val/img` and `rename_val/json` for validation images and annotations
- `rename_testing/img` and `rename_testing/json` for testing images and annotations

### utilities/data2cocoformat.py

Converts all JSON files from training/val/testing into COCO format JSON files

### utilities/GTvisualization.py

Verifies contents of cocoformat_train/val/test.json and visualizes annotations 


## ROBUST-MIPS Dataset 
The **ROBUST-MIPS** dataset and our trained model weights are publicly available for download: 

[🔗 Download ROBUST-MIPS](https://doi.org/10.7303/syn64023381)

## Data Layout

Your dataset directory should look like this:

```text
dataset/
├── training/
│   ├── img/                      # raw training images
│   └── json/                     # any original per-image metadata
├── val/
│   ├── img/                      # raw validation images
│   └── json/                     # any original per-image metadata
├── testing/
│   ├── img/                      # raw test images
│   └── json/                     # any original per-image metadata
├── converted_detections_val.json   # results of mmdetection model on val set
├── converted_detections_test.json  # results of mmdetection model on testing set
├── cocoformat_train.json         # COCO‐style train annotations
├── cocoformat_val.json           # COCO‐style val annotations
└── cocoformat_test.json          # COCO‐style test annotations
```

## tool

Training model:
```
cd /workspace/surgicaltool_bm

python /workspace/surgicaltool_bm/tools/train.py \
       /workspace/surgicaltool_bm/configs/rtmpose-l_8xb256-420e_coco-256x192.py \
       --resume
```

Testing:
```
cd /workspace/surgicaltool_bm

python /workspace/surgicaltool_bm/tools/test.py \
       /workspace/surgicaltool_bm/configs/rtmpose-l_8xb256-420e_coco-256x192.py \
       /workspace/surgicaltool_bm/work_dirs/best_coco_AP_epoch_285.pth \
       --work-dir /workspace/surgicaltool_bm/work_dirs/test/rtmpose-l_8xb256-420e_coco-256x192 \
       --show-dir outputs \
       --out /workspace/surgicaltool_bm/work_dirs/test/rtmpose-l_8xb256-420e_coco-256x192/metric_results.json \
       --cfg-options='model.test_cfg.output_heatmaps=True' \
```
<!-- Should you wish to use or refer to this data set, you must cite the following papers:
1. **Zhe, H.**, **Charlie, B.**, **Gongyu, Z.**, **Huanyu, T.**, **Christos, B.**, **Tom, V.**, *ROBUST-MIPS: A Combined Skeletal Pose Representation and Instance Segmentation Dataset for Laparoscopic Surgical Instruments*, arXiv preprint arXiv:xxxx.xxxx, 2025. Available at: [arXiv link](https://arxiv.org/abs/xxxx.xxxx)
2. **Maier-Hein, L., Wagner, M., Ross, T., Reinke, A., Bodenstedt, S., Full, P. M., ... & Müller-Stich, B. P. (2021)**. *Heidelberg colorectal data set for surgical data science in the sensor operating room*. Scientific data, 8(1), 1-11.
3. **Roß, T., Reinke, A., Full, P. M., Wagner, M., Kenngott, H., Apitz, M., ... & Maier-Hein, L. (2021)**. *Comparative validation of multi-instance instrument segmentation in endoscopy: results of the ROBUST-MIS 2019 challenge*. Medical image analysis, 70, 101920. -->


## Citation

```bibtex

@misc{han2025robustmipscombinedskeletalpose,
      title={ROBUST-MIPS: A Combined Skeletal Pose and Instance Segmentation Dataset for Laparoscopic Surgical Instruments}, 
      author={Zhe Han and Charlie Budd and Gongyu Zhang and Huanyu Tian and Christos Bergeles and Tom Vercauteren},
      year={2025},
      eprint={2508.21096},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.21096}, 
}

```

## License ROBUST-MIPS is realeased unde a Creative Commons Attribution-NonCommercial-ShareAlike (CC BY-NC-SA) license, which means that it will be publicly available for non-commercial usage. 
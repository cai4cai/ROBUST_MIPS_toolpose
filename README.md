# ROBUST-MIPS: A Combined Skeletal Pose and Instance Segmentation Dataset for Laparoscopic Surgical Instruments

This repository provides implementation of surgical-tool pose estimation (based on **ROBUST-MIPS**).
Built on top of [MMPose v1.3.0](https://github.com/open-mmlab/mmpose).

## Introduction
Localisation of surgical tools constitutes a foundational building block for computer-assisted interventional technologies. Works in this field typically focus on training deep learning models to perform segmentation tasks. Performance of learning-based approaches is limited by the availability of diverse annotated data. We argue that skeletal pose annotations are a more efficient annotation approach for surgical tools, striking a balance between richness of semantic information and ease of annotation, thus allowing for accelerated growth of available annotated data. To encourage adoption of this annotation style, we present, ROBUST-MIPS, a combined tool pose and tool instance segmentation dataset derived from the existing ROBUST-MIS dataset. Our enriched dataset facilitates the joint study of these two annotation styles and allow head-to-head comparison on various downstream tasks.
To demonstrate the adequacy of pose annotations for surgical tool localisation, we set up a simple benchmark using popular pose estimation methods and observe high-quality results. To ease adoption, together with the dataset, we release our benchmark models and custom tool pose annotation software.

[🔗 Download ROBUST_MIPS_toolpose](https://github.com/cai4cai/ROBUST_MIPS_toolpose)

[🔗 Download tool-pose-annotation-gui](https://github.com/cai4cai/tool-pose-annotation-gui)

## Overview

This project extends the MMPose v1.3.0 framework to support:

- **Custom dataset**: `ROBUST-MIPS` for laparoscopic frames  
- **Custom evaluation metrics**: custom OKS metric parameters     
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
Should you wish to use or refer to this benchmark, you must cite the following papers:

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
@misc{mmpose2020,
  title        = {OpenMMLab Pose Estimation Toolbox and Benchmark},
  author       = {MMPose Contributors},
  howpublished = {\url{https://github.com/open-mmlab/mmpose}},
  year         = {2020}
}
@article{maier2021heidelberg,
  title={Heidelberg colorectal data set for surgical data science in the sensor operating room},
  author={Maier-Hein, Lena and Wagner, Martin and Ross, Tobias and Reinke, Annika and Bodenstedt, Sebastian and Full, Peter M and Hempe, Hellena and Mindroc-Filimon, Diana and Scholz, Patrick and Tran, Thuy Nuong and others},
  journal={Scientific data},
  volume={8},
  number={1},
  pages={101},
  year={2021},
  publisher={Nature Publishing Group UK London}
}
@article{ross2021comparative,
  title={Comparative validation of multi-instance instrument segmentation in endoscopy: results of the ROBUST-MIS 2019 challenge},
  author={Ro{\ss}, Tobias and Reinke, Annika and Full, Peter M and Wagner, Martin and Kenngott, Hannes and Apitz, Martin and Hempe, Hellena and Mindroc-Filimon, Diana and Scholz, Patrick and Tran, Thuy Nuong and others},
  journal={Medical image analysis},
  volume={70},
  pages={101920},
  year={2021},
  publisher={Elsevier}
}
```

## License ROBUST-MIPS is realeased unde a Creative Commons Attribution-NonCommercial-ShareAlike (CC BY-NC-SA) license, which means that it will be publicly available for non-commercial usage. 
## Optimization of Spatial-Class Alignment in Efficient Semantic Segmentation via Learnable Offsets


### 💡 Core Idea: Optimization via Dynamic Relaxation

**OffSeg** introduces a novel **Offset Learning Paradigm** designed to solve the *optimization bottleneck* in efficient segmentation models.

Traditional methods rely on a **static optimization target** (per-pixel classification with fixed prototypes), which leads to high residual errors when mapping diverse image features to a compressed latent space. OffSeg reformulates this as a **dynamic optimization problem**:

  * **Dual-Branch Optimization**: Explicitly learns `feature offsets` ($\Delta E$) and `class offsets` ($\Delta W$) to relax the rigid constraints of the objective function.
  * **Pareto Efficiency**: Pushes the trade-off frontier between accuracy and computational cost, achieving better convergence with negligible parameter overhead (0.1-0.2M).

*Figure 1: Overview of the Offset Learning framework. Instead of a static projection, we formulate the inference process as a dynamic optimization task where the network predicts optimal offsets to minimize alignment error.*

\<details\>
\<summary\>\<b\>Click to read the Abstract\</b\>\</summary\>

Semantic segmentation is fundamental to vision systems requiring pixel-level scene understanding, yet deploying it on resource-constrained devices demands efficient architectures. Although existing methods achieve real-time inference through lightweight designs, we reveal their inherent limitation: **misalignment between class representations and image features caused by a per-pixel classification paradigm**.

With experimental analysis, we find that this paradigm results in a highly challenging assumption for efficient scenarios: Image pixel features should not vary for the same category in different images. To address this dilemma, we propose a coupled dual-branch offset learning paradigm that explicitly learns feature and class offsets to dynamically refine both class representations and spatial image features. Based on the proposed paradigm, we construct an efficient semantic segmentation network, **OffSeg**. Notably, the offset learning paradigm can be adopted to existing methods with no additional architectural changes.

\</details\>

-----

## ✨ Features

  * **📈 Optimization-Centric Design**: Solves the feature misalignment problem by introducing dynamic slack variables (offsets) into the inference pipeline.
  * **🧩 Plug-and-Play**: Can be seamlessly integrated into existing architectures (e.g., SegFormer, SegNeXt, Mask2Former) as a meta-optimization module.
  * **⚡ Lightweight & Efficient**: Improves convergence and accuracy on ADE20K, Cityscapes, and COCO-Stuff with minimal FLOPs increase.
  * **🏆 State-of-the-Art**: Outperforms existing efficient models by pushing the Pareto frontier of accuracy vs. latency.


## 🛠️ Installation

```bash
conda create -n offseg python=3.9 -y
conda activate offseg

# Install PyTorch (Example for CUDA 11.8)
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install OpenMMLab dependencies
pip install -U openmim
mim install mmengine
mim install mmcv==2.0.0
mim install mmdet

# Install other dependencies
pip install ftfy transformers==4.28.0

# Install OffSeg
pip install -e .
```

## 📂 Data Preparation

Please follow the [MMSegmentation Guidelines](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#prepare-datasets) for preparing datasets. Symlink your dataset root to `OffSeg/data`.

\<details\>
\<summary\>\<b\>Recommended Directory Structure\</b\>\</summary\>

```
OffSeg
├── data
│   ├── ade
│   │   ├── ADEChallengeData2016
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   ├── gtFine
│   ├── coco_stuff164k
│   │   ├── images
│   │   ├── annotations
│   ├── VOCdevkit
```

\</details\>

## 🚀 Usage

### 1\. Training

To train the model and observe the convergence behavior:

```bash
# Single GPU
python tools/train.py local_configs/offseg/Base/offseg-b_ade20k_160k-512x512.py

# Multi-GPU (e.g., 8 GPUs)
bash tools/dist_train.sh local_configs/offseg/Base/offseg-b_ade20k_160k-512x512.py 8
```

### 2\. Evaluation

To evaluate the model efficiency (mIoU vs. FLOPs):

```bash
# OffSeg-T on ADE20K
bash tools/dist_test.sh local_configs/offseg/Tiny/offseg-t_ade20k_160k-512x512.py /path/to/checkpoint.pth 8
```

### 3\. Visualization

Visualize the segmentation results to verify qualitative improvements:

```bash
python tools/test.py local_configs/offseg/Tiny/offseg-t_ade20k_160k-512x512.py /path/to/checkpoint.pth --show-dir ./vis_results
```

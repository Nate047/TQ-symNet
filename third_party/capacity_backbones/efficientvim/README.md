<p align="center">
  <h1 align="middle">EfficientViM</h1>
  </p>
<p align="center">
  <img src="assets/toy_mamba.png" width="300px" />
  <h3 align="middle">EfficientViM: Efficient Vision Mamba with Hidden State Mixer-based State Space Duality</h2>
  <p align="middle">
    <a href="https://www.sanghyeoklee.com/" target="_blank">Sanghyeok Lee</a>, 
    <a href="https://scholar.google.com/citations?user=IaQRhu8AAAAJ&hl=ko" target="_blank">Joonmyung Choi</a>, 
    <a href="https://hyunwoojkim.com/" target="_blank">Hyunwoo J. Kim</a>*
  </p>
  <p align="middle">
    Conference on Computer Vision and Pattern Recognition (CVPR), 2025
  </p>
<!--   <p align="middle">NeurIPS 2024</p> -->
  <p align="middle">
    <a href="https://arxiv.org/abs/2411.15241" target='_blank'><img src="https://img.shields.io/badge/arXiv-2411.15241-b31b1b.svg?logo=arxiv"></a>
  </p>

</p>

---

This repository is an official implementation of the CVPR 2025 paper EfficientViM: Efficient Vision Mamba with Hidden State Mixer-based State Space Duality.

## Main Results
### Comparison of efficient networks on ImageNet-1K classification. 
The family of EfficientViM, marked as red and blue stars, shows the best speed-accuracy trade-offs. ($✝$: with distillation)
<div align="center">
  <img src="assets/comparison.png" width="800px" />
</div>

### Image classification on ImageNet-1K ([pretrained models](https://drive.google.com/drive/folders/1DNoadLbt8GXBGhDgOH9SH4mS4pJ8xfCA?usp=drive_link))
|       model      | resolution | epochs  | acc  | #params | FLOPs |           checkpoint             |
|:---------------:|:----------:|:-----:|:-----:|:-------:|:-----:|:----------------------------:|
| EfficientViM-M1 |  224x224   | 300   | 72.9  |   6.7M  |  239M | [EfficientViM_M1_e300.pth](https://drive.google.com/file/d/1zat4-1npC-kvVkDsCDp--c5Sou7gn5Uj/view?usp=drive_link) |
| EfficientViM-M1 |  224x224   | 450   | 73.5  |   6.7M  |  239M | [EfficientViM_M1_e450.pth](https://drive.google.com/file/d/1ztpSVnWccoGEEobpmk107US3MjX4LEgz/view?usp=drive_link) |
| EfficientViM-M2 |  224x224   | 300   | 75.4  |  13.9M  |  355M | [EfficientViM_M2_e300.pth](https://drive.google.com/file/d/1Mm7Ems4KHFCunNJ8iI0FFQOC3hav7sEj/view?usp=drive_link) |
| EfficientViM-M2 |  224x224   | 450   | 75.8  |  13.9M  |  355M | [EfficientViM_M2_e450.pth](https://drive.google.com/file/d/1_YbrVx06cLtAaCgLG-jKH5tx1mLmnfO2/view?usp=drive_link) |
| EfficientViM-M3 |  224x224   | 300   | 77.6  |  16.6M  |  656M | [EfficientViM_M3_e300.pth](https://drive.google.com/file/d/1zmqmSwl0FHSQHaqjHHGR1XRalPibi_sf/view?usp=drive_link) |
| EfficientViM-M3 |  224x224   | 450   | 77.9  |  16.6M  |  656M | [EfficientViM_M3_e450.pth](https://drive.google.com/file/d/1z8dxkVLPbZdHaran9pDeuMmILsz5xUlf/view?usp=drive_link) |
| EfficientViM-M4 |  256x256   | 300   | 79.4  |  19.6M  | 1111M | [EfficientViM_M4_e300.pth](https://drive.google.com/file/d/1cfRSlvtPaGxMf1N1N_m3e1_IDh7b_OEq/view?usp=drive_link) |
| EfficientViM-M4 |  256x256   | 450   | 79.6  |  19.6M  | 1111M | [EfficientViM_M4_e450.pth](https://drive.google.com/file/d/1rVnX2FT9AVJdU6xSPhEUllXZlhUYv28z/view?usp=drive_link) |

### Image classification on ImageNet-1K with distillation
|       model      | resolution | epochs | acc  |          checkpoint              |
|:---------------:|:----------:|:-----:|:-----:|:----------------------------:|
| EfficientViM-M1 |  224x224   | 300   | 74.6  |[EfficientViM_M1_dist.pth](https://drive.google.com/file/d/1S0HSauwj-d-2_tqf20fBlNAdQbUxhypD/view?usp=drive_link) |
| EfficientViM-M2 |  224x224   | 300   | 76.7  |[EfficientViM_M2_dist.pth](https://drive.google.com/file/d/1HrzY3hI1F0FwXXy8ejO6kXUi-QGK4I74/view?usp=drive_link) |
| EfficientViM-M3 |  224x224   | 300   | 79.1  |[EfficientViM_M3_dist.pth](https://drive.google.com/file/d/13u_UgyOC2sH1ThcQO4GhnW572SggfsTq/view?usp=drive_link) |
| EfficientViM-M4 |  256x256   | 300   | 80.7  |[EfficientViM_M4_dist.pth](https://drive.google.com/file/d/1gvINJ-oszWj7_Gb9wAYuIGV15xWDAQxQ/view?usp=drive_link) |

## Getting Started

### Installation
```bash
# Clone this repository:
git clone https://github.com/mlvlab/EfficientViM.git
cd EfficientViM

# Create and activate the environment
conda create -n EfficientViM python==3.10
conda activate EfficientViM

# Install dependencies
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

### Training
To train EfficientViM for classification on ImageNet, run `train.sh` in [classification](./classification):
```bash
cd classification
sh train.sh <num-gpus> <batch-size-per-gpu> <epochs> <model-name> <imagenet-path> <output-path>
```
For example, to train EfficientViM-M1 for 450 epochs using 8 GPU (with a total batch size of 2048 calculated as `<num-gpus>` $\times$ `<batch-size-per-gpu>`), run:
```bash
sh train.sh 8 256 450 EfficientViM_M1 <imagenet-path> <output-path>
```

### Training with distillation
To train EfficientViM with distillation objective of [DeiT](https://github.com/facebookresearch/deit), run `train_dist.sh` in [classification](./classification):
```bash
sh train_dist.sh <num-gpus> <batch-size-per-gpu> <model-name> <imagenet-path> <output-path>
```

### Evaluation
To evaluate a pre-trained EfficientViM, run `test.sh` in [classification](./classification):
```bash
sh test.sh <num-gpus> <model-name> <imagenet-path> <checkpoint-path>
# For evaluation with the model trained with distillation
# sh test_dist.sh <num-gpus> <model-name> <imagenet-path> <checkpoint-path>
```

## Acknowledgements
This repo is built upon [Swin](https://github.com/microsoft/Swin-Transformer), [VSSD](https://github.com/YuHengsss/VSSD), [SHViT](https://github.com/ysj9909/SHViT), [EfficientViT](https://github.com/microsoft/Cream), and [SwiftFormer](https://github.com/Amshaker/SwiftFormer).  
Thanks to the authors for their inspiring works!

## Citation
If this work is helpful for your research, please consider citing it.
```
@inproceedings{lee2025efficientvim,
  title={EfficientViM: Efficient Vision Mamba with Hidden State Mixer based State Space Duality},
  author={Lee, Sanghyeok and Choi, Joonmyung and Kim, Hyunwoo J.},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```

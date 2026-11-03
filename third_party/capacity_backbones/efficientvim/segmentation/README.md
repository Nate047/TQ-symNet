
# Semantic Segmentation
## Models
|        Model        | mIoU | checkpoint  |      Log       |
|:-------------------:|:----:|:-----------:|:--------------:|
| EfficientViM-M4-450 | 41.3 | [ckpt](https://drive.google.com/file/d/1TUVsEQeZBZDILoDLUUxSkDDRlEMvydvu/view?usp=drive_link) | [log](./logs/efficientViM_ade20k.json) |

## Requirements
Install [mmcv-full](https://github.com/open-mmlab/mmcv) and [MMSegmentation v0.30.0](https://github.com/opedn-mmlab/mmsegmentation/tree/v0.30.0). 
Later versions should work as well. 
The easiest way is to install via [MIM](https://github.com/open-mmlab/mim)
```
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.1/index.html
git clone https://github.com/open-mmlab/mmsegmentation.git -b v0.30.0
cd mmsegmentation
pip install -e .
```

## Data preparation

We benchmark EfficientViM on the challenging ADE20K dataset, which can be downloaded and prepared following [insructions in MMSeg](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#prepare-datasets). 
The data should appear as: 
```
├── segmentation
│   ├── data
│   │   ├── ade
│   │   │   ├── ADEChallengeData2016
│   │   │   │   ├── annotations

│   │   │   │   │   ├── training
│   │   │   │   │   ├── validation
│   │   │   │   ├── images
│   │   │   │   │   ├── training
│   │   │   │   │   ├── validation

```

## Training
To train EfficientViM for segmentation on ade20k, run `./tools/dist_train.sh`:
```
bash ./tools/dist_train.sh configs/sem_fpn/fpn_efficientvim_m4_ade20k_40k.py 8
```


## Evaluation
To evaluate a pre-trained EfficientViM, run `./tools/dist_test.sh`:
```
bash ./tools/dist_test.sh configs/sem_fpn/fpn_efficientvim_m4_ade20k_40k.py path/to/checkpoint #GPUs --eval mIoU
```


## Acknowledge

The downstream task implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.
[repvit](https://github.com/THU-MIG/RepViT/tree/main/segmentation).

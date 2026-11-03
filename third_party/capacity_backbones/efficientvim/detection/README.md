
# Semantic Segmentation
## Results and Models
### RetinaNet Object Detection
|        Model        |  Pretrain   | Box AP  | AP@50 | AP@75 | checkpoint |               Log                      | 
|:-------------------:|:-----------:|:-------:|:-----:|:-----:|:----------:|:--------------------------------------:|
| EfficientViM-M4-450 | ImageNet-1k |  38.8   | 59.6  | 41.1  | [ckpt](https://drive.google.com/file/d/1OrE4pJGVwl8q-kEQghZniIxbBsUi9n9B/view?usp=drive_link) | [log](./logs/efficientViM_retina.json) |


### Mask R-CNN Instance Segmentation
|        Model        | Box AP | Mask AP | checkpoint  |    Log     |
|:-------------------:|:------:|:-------:|:-----------:|:----------:|
| EfficientViM-M4-450 |  39.7  |  36.4   | [ckpt](https://drive.google.com/file/d/17V1DqVFERP5EItGIlaYK0WWYaiTuk4oR/view?usp=drive_link) |  [log](./logs/efficientViM_mask_rcnn.json)   |


## Setup
```
# Create and activate the environment
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.1/index.html
pip install mmdet==2.25.0
```

## Data preparation
Prepare COCO 2017 dataset according to the [instructions in MMDetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md#test-existing-models-on-standard-datasets).
The dataset should be organized as 
```
downstream
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
```


## Training
To train the RetinaNet model with EfficientViM as backbone on a single machine using multi-GPUs, run:

```
bash ./dist_train.sh configs/retinanet_efficientvim_fpn_1x_coco.py 1 --cfg-options model.backbone.pretrained=$PATH_TO_IMGNET_PRETRAIN_MODEL data.samples_per_gpu=16
```

## Evaluation
To evaluate the RetinaNet model with EfficientViM as backbone, run:
```
bash ./dist_test.sh configs/retinanet_efficientvim_fpn_1x_coco.py $PATH_TO_TRAINED_MODEL 1 --eval bbox
```

To evaluate the Mask R-CNN model with EfficientViM as backbone, run:

```
bash ./dist_test.sh configs/mask_rcnn_efficientvim_fpn_1x_coco.py $PATH_TO_TRAINED_MODEL 1 --eval segm
```

## Acknowledge

The downstream task implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.
[SHViT](https://github.com/ysj9909/SHViT).
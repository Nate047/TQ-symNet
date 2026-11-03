#!/bin/bash
nGPUs=$1
MODEL=$2
DATA_PATH=$3
CHECKPOINT=$4

python3 -m torch.distributed.launch --nproc_per_node=$nGPUs --master_addr="127.0.0.1" --master_port=29502 main.py --data-path $DATA_PATH --name $MODEL --resume $CHECKPOINT --eval --distillation-type hard
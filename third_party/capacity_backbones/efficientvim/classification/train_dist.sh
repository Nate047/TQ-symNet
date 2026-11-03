#!/bin/bash
nGPUs=$1
BATCH_SIZE=$2
MODEL=$3
DATA_PATH=$4
OUTPUT=${5:-"output"}

python3 -m torch.distributed.launch --nproc_per_node=$nGPUs --master_addr="127.0.0.1" --master_port=29502 main.py --data-path $DATA_PATH --name $MODEL --batch-size $BATCH_SIZE --output $OUTPUT --distillation-type hard

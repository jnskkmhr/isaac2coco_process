#!/bin/bash
DATA_ROOT=$1

python3 test/test_coco_conversion.py \
    --root_dirs $(ls -d $DATA_ROOT/*) \
    --save_dir $DATA_ROOT \
    --HW $2 $3 \
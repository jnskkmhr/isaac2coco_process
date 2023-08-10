#!/bin/bash

python3 test/test_coco_conversion.py \
    --root_dirs $(ls -d /home/lunar4/jnskkmhr/Lunalab_/Lunalab_standalone/data/*) \
    --save_dir /home/lunar4/jnskkmhr/Lunalab_/Lunalab_standalone/data \
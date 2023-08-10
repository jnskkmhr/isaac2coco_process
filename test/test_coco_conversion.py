import sys
import os
import argparse

sys.path.append(os.getcwd())

from src import COCO_SegDataset, COCO_SegDataset_multi

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dirs", type=str, nargs='+')
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--HW", type=int, nargs=2, default=[480, 640])
    parser.add_argument('--modality', nargs='+', default=["rgb", "object_detection", "instance_segmentation", "instance_segmentation_mapping", "instance_segmentation_semantics_mapping"])
    args = parser.parse_args()
    
    # converter = COCO_SegDataset_multi(args)
    converter = COCO_SegDataset(args)
    converter.run()
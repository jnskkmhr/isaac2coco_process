import sys
import os
import argparse

sys.path.append(os.getcwd())

from src import COCO_SegDataset, COCO_SegDataset_multi

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dirs", type=str, nargs='+')
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--HW", type=int, nargs=2, required=True)
    parser.add_argument('--modality', nargs='+', default=["rgb", "object_detection", "instance_segmentation", "instance_segmentation_mapping", "instance_segmentation_semantics_mapping"])
    args = parser.parse_args()
    converter = COCO_SegDataset(root_dirs=args.root_dirs, save_dir=args.save_dir, modality=args.modality, HW=args.HW)
    converter.run()
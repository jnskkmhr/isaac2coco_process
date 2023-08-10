import os
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool, Value
import functools
import argparse
from glob import glob
from pathlib import Path
import json
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import numpy as np 
from skimage.measure import label, regionprops, find_contours
from skimage.segmentation import find_boundaries
from shapely.geometry import Polygon, MultiPolygon


class BasicDataConverter:
    def __init__(self, args):
        # modality = ["rgb", "depth", "object_detection_npy", "object_detection", "depth_npy", "depth", "semantic_segmentation", "instance_segmentation", "instance_segmentation_mapping"]
        self.root_dirs = args.root_dirs
        self.save_dir = args.save_dir
        self.modality = args.modality
        self.HW = args.HW

    def register_ws(self, root_dir):
        self.root_dir = root_dir
        print(f"Currently working on {root_dir}")


    def read_file_path(self):
        self.file_path = dict()
        if not self.modality:
            pass
        else:
            for mod in self.modality:
                mod_file_path = sorted(glob(os.path.join(self.root_dir, mod, "*")), reverse=False)
                self.file_path.update({mod:mod_file_path})
    
    def make_sub_dirs(self):
        flag = False
        for mod in self.modality:
            if os.path.isdir(os.path.join(self.root_dir, mod)):
                flag = True
                break
            os.makedirs(os.path.join(self.root_dir, mod), exist_ok=True)
        
        if "rgb" in self.modality:
            for rgb_file in glob(os.path.join(self.root_dir, "rgb_*.png")):
                shutil.move(rgb_file, os.path.join(self.root_dir, "rgb", os.path.basename(rgb_file)))
        
        if "depth" in self.modality:
            for depth_file in glob(os.path.join(self.root_dir, "distance_to_image_plane_*.npy")):
                shutil.move(depth_file, os.path.join(self.root_dir, "depth_npy", os.path.basename(depth_file)))
        
        if "object_detection" in self.modality:
            for bbox_file in glob(os.path.join(self.root_dir, "bounding_box_2d_tight_*.npy")):
                shutil.move(bbox_file, os.path.join(self.root_dir, "object_detection", os.path.basename(bbox_file)))
        
        if "semantic_segmentation_color" in self.modality:
            for semantic_file in glob(os.path.join(self.root_dir, "semantic_*.png")):
                shutil.move(semantic_file, os.path.join(self.root_dir, "semantic_segmentation_color", os.path.basename(semantic_file)))
            
            for semantic_file in glob(os.path.join(self.root_dir, "semantic_*.json")):
                os.remove(semantic_file)
        
        if "instance_segmentation" in self.modality:
            for instance_file in glob(os.path.join(self.root_dir, "instance_*.png")):
                shutil.move(instance_file, os.path.join(self.root_dir, "instance_segmentation", os.path.basename(instance_file)))
        
        if "instance_segmentation_mapping" in self.modality:
            for instance_map_file in glob(os.path.join(self.root_dir, "instance_segmentation_mapping_*.json")):
                shutil.move(instance_map_file, os.path.join(self.root_dir, "instance_segmentation_mapping", os.path.basename(instance_map_file)))
        
        if "instance_segmentation_semantics_mapping" in self.modality:
            for instance_sema_map_file in glob(os.path.join(self.root_dir, "instance_segmentation_semantics_mapping_*.json")):
                shutil.move(instance_sema_map_file, os.path.join(self.root_dir, "instance_segmentation_semantics_mapping", os.path.basename(instance_sema_map_file)))
        
        if "object_detection" in self.modality:
            for bbox_prim_file in glob(os.path.join(self.root_dir, "bounding_box_2d_tight_*.json")):
                os.remove(bbox_prim_file)
        return flag
    
    def run(self):
        for root_dir, save_dir in zip(self.root_dirs, self.save_dirs):
            self.register_ws(root_dir, save_dir)
            flag = self.make_sub_dirs()


class Isaac2Basic_Converter(BasicDataConverter):
    """
    RGB, Depth, Semaseg
    """

    def read_npy(self, path:str):
        return np.load(path)

    def rename_file(self, from_path, to_path):
        os.rename(from_path, to_path)
    
    def fetch_data(self):
        depth_path = os.path.join(self.root_dir, "depth_npy") #reference path
        rgb_path = os.path.join(self.root_dir, "rgb")
        semantic_path = os.path.join(self.root_dir, "semantic_segmentation_color")

        depth_savepath = os.path.join(self.save_dir, "depth") # target path
        os.makedirs(depth_savepath, exist_ok=True)
        semaseg_savepath = os.path.join(self.save_dir, "semantic_segmentation")
        filename_list = sorted(os.listdir(rgb_path))

        for i, filename in tqdm(enumerate(filename_list)):
            framename = Path(filename).stem.split('_')[-1]
            depth_file_path = os.path.join(depth_path, 'distance_to_image_plane_'+framename+'.npy')
            rgb_file_path = os.path.join(rgb_path, 'rgb_'+framename+'.png')
            semantic_file_path = os.path.join(semantic_path, 'semantic_segmentation_'+framename+'.png')

            depth_file_savepath = os.path.join(depth_savepath, str(i)+'.png')
            semaseg_file_savepath = os.path.join(semaseg_savepath, str(i)+'.png')

            # convert isaac depth(*.npy) into png
            if os.path.isfile(depth_file_path):
                im = np.load(depth_file_path)
                depth_scale = 100.0
                im = np.clip((im/depth_scale)*255, 0, 255)
                cv2.imwrite(depth_file_savepath, im)

            if os.path.isfile(semantic_file_path):
                mask = cv2.imread(semantic_file_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
                cv2.imwrite(semaseg_file_savepath, mask)
            
            self.rename_file(rgb_file_path, os.path.join(rgb_path, str(i)+'.png'))
            self.rename_file(semantic_file_path, os.path.join(semantic_path, str(i)+'.png'))
        print("Conversion finished")
    
    def run(self):
        for root_dir in self.root_dirs:
            self.register_ws(root_dir)
            flag = self.make_sub_dirs()
            if flag:
                pass
            else:
                self.fetch_data()


class Isaac2Yolo_Det_Converter(BasicDataConverter):

    def read_npy(self, path:str):
        return np.load(path)

    def convert2yolo(self, data:np.ndarray, H:int, W:int):
        """
        Convert bbox in npy to Darknet Yolo txt format
        """
        label = []
        for n in range(data.shape[0]):
            single_box = data[n]
            id = single_box[0]
            xmin = single_box[1]
            ymin = single_box[2]
            xmax = single_box[3]
            ymax = single_box[4]
            w = xmax - xmin
            h = ymax - ymin
            xc = (xmin+xmax)/2
            yc = (ymin+ymax)/2
            scaled_xc = xc/W
            scaled_yc = yc/H
            scaled_w = w/W
            scaled_h = h/H
            label.append([id, scaled_xc, scaled_yc, scaled_w, scaled_h])
        return label

    def save_as_txt(self, data:list, save_path):
        with open(save_path, 'w') as file:
            for row in data:
                file.write(' '.join([str(item) for item in row]))
                file.write('\n')

    def rename_file(self, from_path, to_path):
        os.rename(from_path, to_path)

    
    def fetch_data(self):
        bbox_path = os.path.join(self.root_dir, "object_detection_npy") #reference path
        depth_path = os.path.join(self.root_dir, "depth_npy") #reference path
        rgb_path = os.path.join(self.root_dir, "rgb")
        instance_path = os.path.join(self.root_dir, "instance_segmentation")

        bbox_savepath = os.path.join(self.save_dir, "object_detection") # target path
        depth_savepath = os.path.join(self.save_dir, "depth") # target path
        os.makedirs(bbox_savepath, exist_ok=True)
        os.makedirs(depth_savepath, exist_ok=True)

        filename_list = os.listdir(bbox_path)
        for i, filename in tqdm(enumerate(filename_list)):
            framename = Path(filename).stem.split('_')[-1]

            bbox_file_path = os.path.join(bbox_path, filename)
            depth_file_path = os.path.join(depth_path, 'distance_to_image_plane_'+framename+'.npy')
            rgb_file_path = os.path.join(rgb_path, 'rgb_'+framename+'.png')
            instance_file_path = os.path.join(instance_path, 'instance_segmentation_'+framename+'.png')

            bbox_file_savepath = os.path.join(bbox_savepath, str(i)+'.txt')
            depth_file_savepath = os.path.join(depth_savepath, str(i)+'.png')

            # convert isaac bbox(*.npy) into yolo format(*.txt)
            yolo_bbox = self.convert2yolo(self.read_npy(bbox_file_path), self.args.HW[0], self.args.HW[1])
            self.save_as_txt(yolo_bbox, bbox_file_savepath)

            # convert isaac depth(*.npy) into png
            if os.path.isfile(depth_file_path):
                im = np.load(depth_file_path)
                depth_scale = 20.0
                im = np.clip((im/depth_scale)*255, 0, 255)
                cv2.imwrite(depth_file_savepath, im)
            
            self.rename_file(rgb_file_path, os.path.join(rgb_path, str(i)+'.png'))
            self.rename_file(instance_file_path, os.path.join(instance_path, str(i)+'.png'))
        print("Conversion finished")
    
    def run(self):
        for root_dir in self.root_dirs:
            self.register_ws(root_dir)
            flag = self.make_sub_dirs()
            self.fetch_data()

class Isaac2Yolo_Seg_Converter(Isaac2Yolo_Det_Converter):
    def __init__(self, args):
        # modality = ["rgb", "depth", "object_detection_npy", "object_detection", "depth_npy", "depth", "semantic_segmentation", "instance_segmentation", "instance_segmentation_mapping"]
        self.root_dirs = args.root_dirs
        self.save_dirs = args.save_dirs
        self.modality = args.modality
        self.HW = args.HW
        self.CLASS2ID = {
            "rock_middle":1, 
            "rock_large":2, 
            "UNLABELLED":0,
            "BACKGROUND": 0,
        }
        self.train_split = 0.9
        self.num_scale = 1 # sample image every 10 frames (coz too many images)
        self.depth_scale = 100.0

    def rgba_mask_to_submasks(self, rgba_mask, meta_dict):
        """
        rgba_mask : [H, W, 4]
        meta_dict : dict

        process rgba mask PNG images (multi-class) and produce dict with keys as str pixel value and values as binary masks
        """
        pixel2class = dict()
        pixel2id = dict()
        sub_masks = dict()
        for key, value in meta_dict.items():
            pixel2class[key] = value['class']
            pixel2id[key] = self.CLASS2ID[value['class']]

        h, w, _ = rgba_mask.shape
        for u in range(h):
            for v in range(w):
                pixel_str = str(tuple(rgba_mask[u, v].tolist()))
                if pixel_str in meta_dict.keys():
                    if pixel_str not in sub_masks.keys():
                        if pixel2class[pixel_str] == 'UNLABELLED':
                            continue
                        elif pixel2class[pixel_str] == 'BACKGROUND':
                            continue
                        sub_masks[pixel_str] = np.zeros((h, w), dtype=np.uint8)
                        sub_masks[pixel_str][u, v] = 1
                    else:
                        sub_masks[pixel_str][u, v] = 1
        return sub_masks, pixel2id, pixel2class
    
    def mask_to_polygon(self, mask_image):
        # reference : https://www.immersivelimit.com/create-coco-annotations-from-scratch
        mask_image_pad = np.pad(mask_image, 1, 'constant', constant_values=0) #[H+2, W+2]
        contours = find_contours(mask_image_pad, 0.5, positive_orientation='low')
        segmentations = []
        polygons = []
        for contour in contours:
            # Flip from (row, col) representation to (x, y)
            # and subtract the padding pixel
            for i in range(len(contour)):
                row, col = contour[i]
                contour[i] = (col - 1, row - 1)

            # Make a polygon and simplify it
            poly = Polygon(contour)
            poly = poly.simplify(1.0, preserve_topology=False)
            polygons.append(poly)
            segmentation = np.array(poly.exterior.coords).ravel().tolist()
            segmentations.extend(segmentation)
        
        # Combine the polygons to calculate the bounding box and area
        multi_poly = MultiPolygon(polygons)
        x, y, max_x, max_y = multi_poly.bounds
        width = max_x - x
        height = max_y - y
        bbox = (x, y, width, height)
        area = multi_poly.area
        return segmentations, area, bbox

    def fetch_data(self, split="train"):
        print("="*80)
        print("Fetching data... | mode: {}".format(split))
        print("="*80)
        rgb_path = os.path.join(self.root_dir, "rgb")
        depth_path = os.path.join(self.root_dir, "depth_npy") #reference path
        instance_path = os.path.join(self.root_dir, "instance_segmentation")
        instance_sema_map_path = os.path.join(self.root_dir, "instance_segmentation_semantics_mapping")

        depth_savepath = os.path.join(self.save_dir, "depth") # target path
        annotation_savepath = os.path.join(self.save_dir, "annotation") # target path

        filename_list = sorted(os.listdir(rgb_path)) # sort by index
        
        for i in tqdm(range(0, len(filename_list), self.num_scale)):
            annotation_list = []
            framename = Path(filename_list[i]).stem.split('_')[-1]
            if len(framename) < 5:
                framename = framename.zfill(4)

            rgb_file_path = os.path.join(rgb_path, 'rgb_'+framename+'.png')
            depth_file_path = os.path.join(depth_path, 'distance_to_image_plane_'+framename+'.npy')
            instance_file_path = os.path.join(instance_path, 'instance_segmentation_'+str(framename)+'.png')
            instance_sema_map_file_path = os.path.join(instance_sema_map_path, 'instance_segmentation_semantics_mapping_'+framename+'.json')

            instance_image = cv2.cvtColor(cv2.imread(instance_file_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
            instance_sema_meta = json.load(open(instance_sema_map_file_path, 'r'))

            sub_masks, pixel2id, pixel2class = self.rgba_mask_to_submasks(instance_image, instance_sema_meta)
            for pixel_str in sub_masks.keys():
                category_id = pixel2id[pixel_str]
                assert category_id > 0, "category_id must be greater than 0"
                segmentations, _, _ = self.mask_to_polygon(sub_masks[pixel_str])
                annotation_list.append([category_id] + segmentations)
            
            annotation_file_savepath = os.path.join(annotation_savepath, str(i)+'.txt')
            depth_file_savepath = os.path.join(depth_savepath, str(i)+'.png')
            self.save_as_txt(annotation_list, annotation_file_savepath)
            
            # convert isaac depth(*.npy) into png
            if os.path.isfile(depth_file_path):
                im = np.load(depth_file_path)
                im = np.clip((im/self.depth_scale)*255, 0, 255)
                cv2.imwrite(depth_file_savepath, im)
            
            self.rename_file(rgb_file_path, os.path.join(rgb_path, str(i)+'.png'))
        
        print("Conversion finished")


class CleanData(BasicDataConverter):
    def clean(self):
        """
        Remove images and respective annotation if image is too dark (meaning it in complete shadow)
        """
        rgb_path_list = self.file_path["rgb"]
        for i, rgb_path in enumerate(rgb_path_list):
            image = cv2.imread(rgb_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            occulusion_rate = 0
            height, width = image.shape
            for u in range(height):
                for v in range(width):
                    if image[u, v] < 1:
                        occulusion_rate += 1
            occulusion_rate = occulusion_rate/(height*width)
            if occulusion_rate > 0.95:
                for mod_paths in self.file_path.values():
                    print("Removing", mod_paths[i])
                    os.remove(mod_paths[i])
    
    def run(self):
        for root_dir in self.root_dirs:
            self.register_ws(root_dir)
            self.read_file_path()
            self.clean()

RGB_TAG = "rgb"
INSTANCE_SEGMENTATION_TAG = "instance_segmentation"
INSTANCE_SEGMENTATION_SEMANTICS_MAPPING_TAG = "instance_segmentation_id_semantic"
# INSTANCE_SEGMENTATION_SEMANTICS_MAPPING_TAG = "instance_segmentation_semantics_mapping"

class COCO_SegDataset(BasicDataConverter):
    def __init__(self, args):
        # modality = ["rgb", "depth", "object_detection_npy", "object_detection", "depth_npy", "depth", "semantic_segmentation", "instance_segmentation", "instance_segmentation_mapping"]
        self.root_dirs = args.root_dirs
        self.save_dir = args.save_dir
        self.modality = args.modality
        self.HW = args.HW

        self.CLASS2ID = {
            "rock":1, 
            "ground":0, 
            "UNLABELLED":0,
            "BACKGROUND": 0,
        }
        self.train_split = 0.95

        self.image_info_list = []
        self.annotation_info_list = []
        self.category_info_list = []
        self.image_id = 0 # counter for image
        self.anno_id = 0 # counter for annotation

    def rgba_mask_to_submasks(self, rgba_mask, meta_dict):
        """
        rgba_mask : [H, W, 4]
        meta_dict : dict

        process rgba mask PNG images (multi-class) and produce dict with keys as str pixel value and values as binary masks
        """
        exclude_class = ['UNLABELLED', 'BACKGROUND', 'ground']
        pixel2class = dict()
        pixel2id = dict()
        sub_masks = dict()
        for key, value in meta_dict.items():
            pixel2class[key] = value['class']
            pixel2id[key] = self.CLASS2ID[value['class']]

        h, w, _ = rgba_mask.shape
        for u in range(h):
            for v in range(w):
                pixel_str = str(tuple(rgba_mask[u, v].tolist()))
                if pixel_str in meta_dict.keys():
                    if pixel_str not in sub_masks.keys():
                        if pixel2class[pixel_str] in exclude_class:
                            continue
                        sub_masks[pixel_str] = np.zeros((h, w), dtype=np.uint8)
                        sub_masks[pixel_str][u, v] = 1
                    else:
                        sub_masks[pixel_str][u, v] = 1
        return sub_masks, pixel2id, pixel2class

    def mask_to_polygon(self, mask_image):
        # reference : https://www.immersivelimit.com/create-coco-annotations-from-scratch
        mask_image_pad = np.pad(mask_image, 1, 'constant', constant_values=0) #[H+2, W+2]
        contours = find_contours(mask_image_pad, 0.5, positive_orientation='low')
        segmentations = []
        polygons = []
        skip = False
        for contour in contours:
            # Flip from (row, col) representation to (x, y)
            # and subtract the padding pixel
            for i in range(len(contour)):
                row, col = contour[i]
                contour[i] = (col - 1, row - 1)

            # Make a polygon and simplify it
            poly = Polygon(contour)
            poly = poly.simplify(1.0, preserve_topology=False)
            polygons.append(poly)
            segmentation = np.array(poly.exterior.coords).ravel().tolist()
            # BUG : do not include empty segmentation list into segmentations (will cause error in COCO API)
            if len(segmentation) > 0:
                segmentations.append(segmentation)
        if not segmentations:
            print("no polygon included in the mask")
            skip = True
        # Combine the polygons to calculate the bounding box and area
        multi_poly = MultiPolygon(polygons)
        x, y, max_x, max_y = multi_poly.bounds
        width = max_x - x
        height = max_y - y
        bbox = (x, y, width, height)
        area = multi_poly.area
        iscrowd = 0
        return segmentations, area, bbox, iscrowd, skip
    
    def create_image_info(self, filename, width, height, image_id):
        images = {
            "file_name": filename,
            "height": height,
            "width": width,
            "id": image_id
        }
        return images
    
    def create_annotation_info(self, polygon, area, bbox, image_id, category_id, annotation_id, iscrowd=0):
        annotation = {
            "segmentation": polygon,
            "area": area,
            "iscrowd": iscrowd,
            "image_id": image_id,
            "bbox": bbox,
            "category_id": category_id,
            "id": annotation_id
        }
        return annotation
    
    def create_category_info(self, category_id, category_name):
        category = {
            "supercategory": category_name,
            "id": category_id,
            "name": category_name
        }
        return category
    
    def get_coco_json_format(self, images, annotations, categories):
        coco_format = {
            "info": {},
            "licenses": {},
            "images": images,
            "annotations": annotations, 
            "categories": categories,
        }
        return coco_format
    
    def fetch_data(self, split="train"):
        print("="*80)
        print("Fetching data... | mode: {}".format(split))
        print("="*80)
        self.rgb_path = os.path.join(self.root_dir, RGB_TAG)
        self.instance_path = os.path.join(self.root_dir, INSTANCE_SEGMENTATION_TAG)
        self.instance_sema_map_path = os.path.join(self.root_dir, INSTANCE_SEGMENTATION_SEMANTICS_MAPPING_TAG)

        if split == "train":
            filename_list = sorted(os.listdir(self.rgb_path)) # sort by index
            train_split = len(filename_list) * self.train_split
            filename_list = filename_list[:int(train_split)]
        elif split == "val":
            filename_list = sorted(os.listdir(self.rgb_path))
            train_split = len(filename_list) * self.train_split
            filename_list = filename_list[int(train_split):]
        
        for filename in tqdm(filename_list):
            self.process_annotation(filename)
        
    def process_annotation(self, filename):
        framename = Path(filename).stem
        image_name = os.path.join(self.root_dir.split('/')[-1], RGB_TAG, framename+".png")
        instance_file_path = os.path.join(self.instance_path, framename+'.png')
        instance_sema_map_file_path = os.path.join(self.instance_sema_map_path, framename+'.json')

        instance_image = cv2.cvtColor(cv2.imread(instance_file_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
        instance_sema_meta = json.load(open(instance_sema_map_file_path, 'r'))

        sub_masks, pixel2id, pixel2class = self.rgba_mask_to_submasks(instance_image, instance_sema_meta)
        for pixel_str in sub_masks.keys():
            category_id = pixel2id[pixel_str]
            assert category_id > 0, "category_id must be greater than 0"
            sub_mask = sub_masks[pixel_str]
            segmentations, area, bbox, iscrowd, skip = self.mask_to_polygon(sub_mask)
            if not skip:
                image_info = self.create_image_info(image_name, self.HW[1], self.HW[0], self.image_id)
                annotation_info = self.create_annotation_info(segmentations, area, bbox, self.image_id, category_id, self.anno_id, iscrowd)
                self.image_info_list.append(image_info)
                self.annotation_info_list.append(annotation_info)
                self.anno_id += 1
        self.image_id += 1
        
    def export_to_json(self, split):
        for category_name, category_id in self.CLASS2ID.items():
            if category_id == 0:
                continue
            category_info = self.create_category_info(category_id, category_name)
            self.category_info_list.append(category_info)

        coco_anotation = self.get_coco_json_format(self.image_info_list, self.annotation_info_list, self.category_info_list)
        with open(os.path.join(self.save_dir, f'instances_{split}2017.json'), 'w') as f:
            json.dump(coco_anotation, f)
        print("Conversion finished")
        self.annotation_info_list = []
        self.image_info_list = []
        self.category_info_list = []

    def run(self):
        for split in ["train", "val"]:
            for root_dir in self.root_dirs:
                self.register_ws(root_dir)
                self.fetch_data(split)
            self.export_to_json(split)



class COCO_SegDataset_multi(BasicDataConverter):
    def __init__(self, args):
        # modality = ["rgb", "depth", "object_detection_npy", "object_detection", "depth_npy", "depth", "semantic_segmentation", "instance_segmentation", "instance_segmentation_mapping"]
        self.root_dirs = args.root_dirs
        self.save_dir = args.save_dir
        self.modality = args.modality
        self.HW = args.HW

        self.CLASS2ID = {
            "rock":1, 
            "ground":0, 
            "UNLABELLED":0,
            "BACKGROUND": 0,
        }
        self.train_split = 0.95

        self.image_info_list = []
        self.annotation_info_list = []
        self.category_info_list = []
        self.image_id = Value('i', 0) # counter for image
        self.anno_id = Value('i', 0) # counter for annotation

    def rgba_mask_to_submasks(self, rgba_mask, meta_dict):
        """
        rgba_mask : [H, W, 4]
        meta_dict : dict

        process rgba mask PNG images (multi-class) and produce dict with keys as str pixel value and values as binary masks
        """
        exclude_class = ['UNLABELLED', 'BACKGROUND', 'ground']
        pixel2class = dict()
        pixel2id = dict()
        sub_masks = dict()
        for key, value in meta_dict.items():
            pixel2class[key] = value['class']
            pixel2id[key] = self.CLASS2ID[value['class']]

        h, w, _ = rgba_mask.shape
        for u in range(h):
            for v in range(w):
                pixel_str = str(tuple(rgba_mask[u, v].tolist()))
                if pixel_str in meta_dict.keys():
                    if pixel_str not in sub_masks.keys():
                        if pixel2class[pixel_str] in exclude_class:
                            continue
                        sub_masks[pixel_str] = np.zeros((h, w), dtype=np.uint8)
                        sub_masks[pixel_str][u, v] = 1
                    else:
                        sub_masks[pixel_str][u, v] = 1
        return sub_masks, pixel2id, pixel2class

    def mask_to_polygon(self, mask_image):
        # reference : https://www.immersivelimit.com/create-coco-annotations-from-scratch
        mask_image_pad = np.pad(mask_image, 1, 'constant', constant_values=0) #[H+2, W+2]
        contours = find_contours(mask_image_pad, 0.5, positive_orientation='low')
        segmentations = []
        polygons = []
        skip = False
        for contour in contours:
            # Flip from (row, col) representation to (x, y)
            # and subtract the padding pixel
            for i in range(len(contour)):
                row, col = contour[i]
                contour[i] = (col - 1, row - 1)

            # Make a polygon and simplify it
            poly = Polygon(contour)
            poly = poly.simplify(1.0, preserve_topology=False)
            polygons.append(poly)
            segmentation = np.array(poly.exterior.coords).ravel().tolist()
            # BUG : do not include empty segmentation list into segmentations (will cause error in COCO API)
            if len(segmentation) > 0:
                segmentations.append(segmentation)
        if not segmentations:
            print("no polygon included in the mask")
            skip = True
        # Combine the polygons to calculate the bounding box and area
        multi_poly = MultiPolygon(polygons)
        x, y, max_x, max_y = multi_poly.bounds
        width = max_x - x
        height = max_y - y
        bbox = (x, y, width, height)
        area = multi_poly.area
        iscrowd = 0
        return segmentations, area, bbox, iscrowd, skip
    
    def create_image_info(self, filename, width, height, image_id):
        images = {
            "file_name": filename,
            "height": height,
            "width": width,
            "id": image_id
        }
        return images
    
    def create_annotation_info(self, polygon, area, bbox, image_id, category_id, annotation_id, iscrowd=0):
        annotation = {
            "segmentation": polygon,
            "area": area,
            "iscrowd": iscrowd,
            "image_id": image_id,
            "bbox": bbox,
            "category_id": category_id,
            "id": annotation_id
        }
        return annotation
    
    def create_category_info(self, category_id, category_name):
        category = {
            "supercategory": category_name,
            "id": category_id,
            "name": category_name
        }
        return category
    
    def get_coco_json_format(self, images, annotations, categories):
        coco_format = {
            "info": {},
            "licenses": {},
            "images": images,
            "annotations": annotations, 
            "categories": categories,
        }
        return coco_format
    
    def fetch_data(self, split="train"):
        print("="*80)
        print("Fetching data... | mode: {}".format(split))
        print("="*80)
        self.rgb_path = os.path.join(self.root_dir, RGB_TAG)
        self.instance_path = os.path.join(self.root_dir, INSTANCE_SEGMENTATION_TAG)
        self.instance_sema_map_path = os.path.join(self.root_dir, INSTANCE_SEGMENTATION_SEMANTICS_MAPPING_TAG)

        if split == "train":
            filename_list = sorted(os.listdir(self.rgb_path)) # sort by index
            train_split = len(filename_list) * self.train_split
            filename_list = filename_list[:int(train_split)]
        elif split == "val":
            filename_list = sorted(os.listdir(self.rgb_path))
            train_split = len(filename_list) * self.train_split
            filename_list = filename_list[int(train_split):]
        
        with Pool(8) as p:
            result = p.map(self.process_annotation, (tqdm(filename_list), self.image_id, self.anno_id))
        
        for r in result:
            if r[0] is not None:
                self.image_info_list.append(r[0])
                self.annotation_info_list.append(r[1])
        for category_name, category_id in self.CLASS2ID.items():
            if category_id == 0:
                continue
            category_info = self.create_category_info(category_id, category_name)
            self.category_info_list.append(category_info)
        
    def process_annotation(self, filename, image_id, anno_id):
        image_info = None
        annotation_info = None
        framename = Path(filename).stem
        image_name = os.path.join(self.root_dir.split('/')[-1], RGB_TAG, framename+".png")
        instance_file_path = os.path.join(self.instance_path, framename+'.png')
        instance_sema_map_file_path = os.path.join(self.instance_sema_map_path, framename+'.json')

        instance_image = cv2.cvtColor(cv2.imread(instance_file_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
        instance_sema_meta = json.load(open(instance_sema_map_file_path, 'r'))

        sub_masks, pixel2id, pixel2class = self.rgba_mask_to_submasks(instance_image, instance_sema_meta)
        for pixel_str in sub_masks.keys():
            category_id = pixel2id[pixel_str]
            assert category_id > 0, "category_id must be greater than 0"
            sub_mask = sub_masks[pixel_str]
            segmentations, area, bbox, iscrowd, skip = self.mask_to_polygon(sub_mask)
            if not skip:
                image_info = self.create_image_info(image_name, self.HW[1], self.HW[0], image_id.value)
                annotation_info = self.create_annotation_info(segmentations, area, bbox, image_id.value, category_id, anno_id.value, iscrowd)
                anno_id.value += 1
            else:
                pass
        image_id.value += 1
        return image_info, annotation_info
        
    def export_to_json(self, split):
        coco_anotation = self.get_coco_json_format(self.image_info_list, self.annotation_info_list, self.category_info_list)
        with open(os.path.join(self.save_dir, f'instances_{split}2017.json'), 'w') as f:
            json.dump(coco_anotation, f)
        print("Conversion finished")
        self.annotation_info_list = []
        self.image_info_list = []
        self.category_info_list = []

    def run(self):
        for split in ["train", "val"]:
            for root_dir in self.root_dirs:
                self.register_ws(root_dir)
                self.fetch_data(split)
            self.export_to_json(split)
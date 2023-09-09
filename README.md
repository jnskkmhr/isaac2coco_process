# Code for converting isaac-sim replicator data into MS COCO json
Synthetic data produced from Isaac sim replicator cannot be used directly by object segmentation libraries. <br>
This is because semantic/instance masks are given in RGBA raster images. <br>
This repo converts PNG mask to bounding polygons (Microsoft COCO json)

## Expected input data structure
The following is the data structure expected. <br>
Each directories with unique hash id contains 4 sub-directories: 
- `rgb` : RGBA PNG image
- `instance_segmentation` : RGBA PNG mask
- `instance_segmentation_id_semantic` : json files to map each mask segment to certain semantics(class)
- `instance_segmentation_id_label` : not used
```

{/path/to/data/root}
├── HI2KckNS14rZBJAu
│   ├── instance_segmentation
│   ├── instance_segmentation_id_label
│   ├── instance_segmentation_id_semantic
│   └── rgb
├── jNPbGDgv58rsMaAE
│   ├── instance_segmentation
│   ├── instance_segmentation_id_label
│   ├── instance_segmentation_id_semantic
│   └── rgb
├── kuwHDzSsW1tvZfYe
│   ├── instance_segmentation
│   ├── instance_segmentation_id_label
│   ├── instance_segmentation_id_semantic
│   └── rgb
└── Yc7xvNabWEIDozdr
    ├── instance_segmentation
    ├── instance_segmentation_id_label
    ├── instance_segmentation_id_semantic
    └── rgb
```

## Usage
Run the following 
```
bash script/test_coco.sh {/path/to/dataset/root} {image_height} {image_width}
```
This will produce `instances_train2017.json` and `instances_val2017.json` in the same directory as {/path/to/dataset/root}. 

## WIP : acceleration with multi-threading
WIP
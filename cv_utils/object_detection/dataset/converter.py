import argparse
import json
import os
import shutil

from copy import deepcopy
from pathlib import Path

from .utils import read_json

def coco_to_yolo(coco_annotation_path, coco_image_folder,
                 output_folder, output_set_name):
    """COCO --> YOLO
    
    - {output_folder}
        - images
            - {output_set_name}
                - {image_1}
                - {image_2}
                - ...
        - labels
            - {output_set_name}
                - {image_1}
                - {image_2}
                - ...

    Args:
        coco_annotation_path (string): [description]
        coco_image_folder (string): [description]
        output_folder (string): [description]
        output_set_name (string): Set name output
    """
    
    # Create the labels and images folder as in the YOLO format
    for name in ["labels", "images"]:
        target_dir = os.path.join(output_folder, name, output_set_name)
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        for path in os.listdir(target_dir):
            try: os.remove(os.path.join(target_dir, path))
            except IsADirectoryError : shutil.rmtree(os.path.join(target_dir, path))
    
    # Read a COCO annotation file
    coco_annotation = read_json(coco_annotation_path)
    
    # Create a dictionary having an image_id as the key
    mapping = {image["id"]: {'file_name': image['file_name'],
                             'height': image['height'],
                             'width': image['width'],
                             'annotations': []}
               for image in coco_annotation['images']}
    
    # Loop over annotation
    for annotation in coco_annotation['annotations']:
        # Get the x, y, w, h value from an annotation
        x, y, w, h = annotation['bbox']
        
        # Get the image_id of an annotation
        image_id = annotation['image_id']
        
        # Get the width and height of the corresponding image
        width = mapping[image_id]['width']
        height = mapping[image_id]['height']
        
        # Get the category_id of an annotation
        # COCO idx starts from 1 whereas YOLO idx starts from 0
        category_id = annotation['category_id'] - 1
        
        # Change bounding-box into as in YOLO format
        xc = (x + w/2) / width
        yc = (y + h/2) / height
        wn = w / width
        hn = h / height
        
        # Append annotation
        mapping[image_id]['annotations'].append([category_id, xc, yc, wn, hn])
        
    # Save into the YOLO format
    # Loop over image
    for _, image in mapping.items():
        # Copy image
        shutil.copy2(os.path.join(coco_image_folder, image['file_name']),
                     os.path.join(output_folder, "images", output_set_name, image['file_name']))
        
        # Save the annotation to a .txt file
        annotations = list(map(lambda x: " ".join(list(map(str, x))), image['annotations']))
        with open(os.path.join(output_folder, "labels", output_set_name,
                               os.path.splitext(image['file_name'])[0]+'.txt'), 'w') as f:
            f.writelines(annotations)
    
    # Get the category
    categories = [None] * len(coco_annotation["categories"])
    for cat in coco_annotation["categories"]:
        categories[cat['id'] - 1] = cat['name']
    
    print("[INFO] Conversion from COCO object detection format to YOLO complete ...")
    print("[INFO] You need to ADD the following list to a .yaml or .txt file")
    print("[INFO] names: {}".format(categories))
    print('-' * 75)
    print('')
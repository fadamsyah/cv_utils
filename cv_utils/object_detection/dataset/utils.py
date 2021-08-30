import json
import os
import shutil

from copy import deepcopy
from pathlib import Path

def create_and_overwrite_dir(path_dir):
    # Create the directory
    Path(path_dir).mkdir(parents=True, exist_ok=True)
    
    # Overwrite the directory
    for path in os.listdir(path_dir):
        try: os.remove(os.path.join(path_dir, path))
        except IsADirectoryError: shutil.rmtree(os.path.join(path_dir, path))

def read_json(path):
    """
    Read a .json file

    Args:
        path (string): Path of a .json file

    Returns:
         data (dictionary): Output dictionary
    """
    
    f = open(path,)
    data = json.load(f)
    f.close()
    
    return data

def write_json(files, path, indent=4):
    """
    Write a json file from a dictionary

    Args:
        files (dictionary): Data
        path (string): Saved json path
        indent (int, optional): Number of spaces of indentation. Defaults to 4.
    """
    
    json_object = json.dumps(files, indent = indent) 

    # Writing to saved_path_json
    with open(path, "w") as outfile: 
        outfile.write(json_object) 
        

        
def coco_to_img2annots(coco_annotations):
    
    # Initialize img2annots
    img2annots = {}
    
    # Generate img2annots key
    num_obj_init = {category['id']: 0 for category in coco_annotations['categories']}
    for image in coco_annotations['images']:
        image_id = image['id']
        img2annots[image_id] = {
            'description': deepcopy(image),
            'annotations': [],
            'num_objects': deepcopy(num_obj_init)
        }
        
    # Add every annotation to its corresponding image key
    for annotation in coco_annotations['annotations']:
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        img2annots[image_id]['annotations'].append(annotation)
        img2annots[image_id]['num_objects'][category_id] += 1
    
    
    return img2annots
    
def yolo_to_img2annots(yolo_annotations, yolo_classes):
    
    pass
    
    # return img2annots
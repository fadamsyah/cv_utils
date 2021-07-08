import math
import os
import random

from copy import deepcopy
from pathlib import Path

from .utils import read_json, write_json
from .utils import coco_to_img2annots

def split_coco(ori_annotation_path, split_dictionary, out_annotation_dir,
               max_iter=100, seed=42):
    '''
    Split dataset according to the split_dictionary
    Optimization objective is minimizing a Sum-Squared Error
    '''
    
    # Read the annotation
    input_annotations = read_json(ori_annotation_path)
    
    # Convert to img2annots
    img2annots = coco_to_img2annots(input_annotations)
    
    # Generate a class_dictionary
    class_dictionary = {category["id"]: category["name"]
                        for category in input_annotations["categories"]}

    # Split the dataset
    img_ids_seq, split_img_dict = split_dataset(
        img2annots, class_dictionary, split_dictionary, max_iter, seed
    )
    
    # Split the dataset according to img_ids_seq
    annotations = {}
    for set_name, start_stop_idx in split_img_dict.items():
        obj_dict = {img_id: img2annots[img_id]
                    for img_id in img_ids_seq[start_stop_idx[0]:start_stop_idx[1]]}
        annotations[set_name] = {
            'type': input_annotations['type'],
            'categories': input_annotations['categories'],
            'images': [],
            'annotations': []
        }
        for _, val2 in obj_dict.items():
            annotations[set_name]['images'].append(val2['description'])
            annotations[set_name]['annotations'].extend(val2['annotations'])
    
    # Save the new annotations
    Path(out_annotation_dir).mkdir(parents=True, exist_ok=True)
    for set_name, dictionary in annotations.items():
        write_json(dictionary,
                   os.path.join(out_annotation_dir, f"instances_{set_name}.json"))

def split_dataset(img2annots, class_dictionary, split_dictionary,
                  max_iter=100, seed=42):
    
    # Normalize the split dictionary, i.e. sum of all fractions must be 1
    total = sum([val for _, val in split_dictionary.items()])
    split_dict = {key: val/total for key, val in split_dictionary.items()}
    
    # Initialize the number of objects of each class
    total_objects = {cat_id: 0 for cat_id in class_dictionary.keys()}
    
    # Calculate the total number of objects of every class
    for key, val in img2annots.items():
        for cat_id, cat_n in val['num_objects'].items():
            total_objects[cat_id] = total_objects[cat_id] + cat_n
            
    # Get the spit_size for every set
    total_img = len(img2annots.keys())
    split_size_img = {}
    for key1, val1 in split_dict.items():
        split_size_img[key1] = math.ceil(val1*total_img)
        
    # Get the index_mapping for each set
    split_img_dict = {}
    start_idx = 0
    for key, val in split_size_img.items():
        split_img_dict[key] = [start_idx, min(start_idx + val, total_img)]
        start_idx = start_idx + val
        
    # Calculate the percentage of objects w.r.t to total objects
    def calculate_object(data_dict, total_objects):
        count = {key: 0 for key, _ in total_objects.items()}
        for key, val in data_dict.items():
            for ann in val['annotations']:
                category_id = ann['category_id']
                count[category_id] = count[category_id] + 1
        for key, val in total_objects.items():
            count[key] = count[key] / val
        return count
    
    # Optimization
    img_ids = list(img2annots.keys())
    obj_counts = {}
    best_error = 1.
    random.seed(seed)
    for i in range(max_iter):
        # Shuffle the img_ids
        random.shuffle(img_ids)
        
        # Calculate total objects for each class in every set
        for key, val in split_img_dict.items():
            obj_dict = {name: img2annots[name]
                        for name in img_ids[val[0]:val[1]]}
            obj_counts[key] = calculate_object(obj_dict, total_objects)
            
        # Calculate the sum-squared-error
        error = 0
        for key1, val1 in split_dictionary.items():
            for _, val2 in obj_counts[key1].items():
                error = error + (val1-val2)**2
        
        # Update the parameters if get a better result
        if error < best_error:
            best_error = deepcopy(error)
            best_img_ids_seq = deepcopy(img_ids)
            print(f'The best error: {best_error}')
            
    return best_img_ids_seq, split_img_dict
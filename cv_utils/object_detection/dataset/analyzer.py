import json
import os

from .utils import read_json

def coco_dataset_analysis(path_annotation_dictionary):
    '''
    Example:
        - path_annotation_dictionary ={
              'set_name_1': 'coco_annotation_1.json',
              'set_name_2': 'coco_annotation_2.json',
              ... : ...,
              'set_name_n': 'coco_annotation_n.json',
          }
    '''
    
    annotations ={
        set_name: read_json(path) for set_name, path
        in path_annotation_dictionary.items()
    }
    
    dataset_analysis(annotations)

def dataset_analysis(annotations, display=True):
    '''
    Example:
        - annotations = {
            'set_name_1': coco_formatted_annotation,
            'set_name_2': coco_formatted_annotation,
            ... : ... ,
            'set_name_n': coco_formatted_annotation,
          }
    '''
    
    set_names = list(annotations.keys())
    categories = annotations[set_names[0]]['categories']
    num_objects = 0
    num_images = 0
    
    results = {set_name: {
        'num_images': 0,
        'num_objects': 0,
        'objects': {category['id']: 0 for category in categories}
    } for set_name in set_names}
    
    for set_name in set_names:
        anns = annotations[set_name]
        for image in anns['images']:
            num_images = num_images + 1
            results[set_name]['num_images'] = results[set_name]['num_images'] + 1
        
        for objs in anns['annotations']:
            cat_id = objs['category_id']
            num_objects = num_objects + 1
            results[set_name]['num_objects'] = results[set_name]['num_objects'] + 1
            results[set_name]['objects'][cat_id] = results[set_name]['objects'][cat_id] + 1
    
    if display:
        print('-----------------------------------')
        print('num_images', ' '*(20 - len(f'num_images{num_images}')), num_images)
        print('num_objects', ' '*(20 - len(f'num_objects{num_objects}')), num_objects)

        print('-----------------------------------')
        print('num_images on each set')
        print('')
        total = sum([results[set_name]['num_images'] for set_name in set_names])
        for set_name in set_names:
            nimgs = results[set_name]['num_images']
            pct = nimgs / total
            print(set_name, ' '*(15-len(f'{set_name}{nimgs}')), nimgs, ' '*2, "{:.3f}".format(pct))

        print('-----------------------------------')
        print('num_objects on each set')
        print('')
        total = sum([results[set_name]['num_objects'] for set_name in set_names])
        for set_name in set_names:
            nobjs = results[set_name]['num_objects']
            pct = nobjs / total
            print(set_name, ' '*(15-len(f'{set_name}{nobjs}')), nobjs, ' '*2, "{:.3f}".format(pct))

        for category in categories:
            cat_id = category['id']
            print('-----------------------------------')
            print(f'Category: {cat_id}')
            print('')
            total = sum([results[set_name]['objects'][cat_id] for set_name in set_names])
            for set_name in set_names:
                nobjs = results[set_name]['objects'][cat_id]
                pct = nobjs / total
                print(set_name, ' '*(15-len(f'{set_name}{nobjs}')), nobjs, ' '*2, "{:.3f}".format(pct))
        print('-----------------------------------')
            
    return results
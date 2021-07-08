from cv_utils.object_detection.dataset.analyzer import coco_dataset_analysis

path_annotation_dictionary = {
    set_name: f"test/ki67/instances_{set_name}.json"
    for set_name in ['train', 'val', 'test']
}

coco_dataset_analysis(path_annotation_dictionary)
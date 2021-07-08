from cv_utils.object_detection.dataset.analyzer import coco_dataset_analysis
from cv_utils.object_detection.dataset.analyzer import yolo_dataset_analysis

path_annotation_dictionary = {
    set_name: f"test/ki67/instances_{set_name}.json"
    for set_name in ['train', 'val', 'test']
}

coco_dataset_analysis(path_annotation_dictionary)

yolo_label_dir = {
    "train": "test/dataset/labels/train",
    "val": "test/dataset/labels/val",
    "test": "test/dataset/labels/test",
}
yolo_class_path = "test/dataset/classes.txt"
yolo_dataset_analysis(yolo_class_path, yolo_label_dir)
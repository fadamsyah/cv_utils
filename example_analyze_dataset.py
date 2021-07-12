from cv_utils.object_detection.dataset.analyzer import coco_dataset_analysis
from cv_utils.object_detection.dataset.analyzer import yolo_dataset_analysis

coco_dataset_analysis(
    {
        "train": "demo/dataset/fasciola_coco/annotations/instances_train.json",
        "val": "demo/dataset/fasciola_coco/annotations/instances_val.json",
        "test": "demo/dataset/fasciola_coco/annotations/instances_test.json",
    }
)

yolo_dataset_analysis(
    "demo/dataset/fasciola_yolo/classes.txt",
    {
        "train": "demo/dataset/fasciola_yolo/labels/train",
        "val": "demo/dataset/fasciola_yolo/labels/val",
        "test": "demo/dataset/fasciola_yolo/labels/test",
    }
)
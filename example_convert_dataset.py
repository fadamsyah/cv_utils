from cv_utils.object_detection.dataset.converter import coco_to_yolo
from cv_utils.object_detection.dataset.converter import yolo_to_coco

''' COCO --> YOLO
This code uses the ultralytics/yolov5 format.
The converted dataset will be saved as follow:
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
    - classes.txt
'''

for set_name in ["train", "test", "val"]:
    coco_to_yolo(
        coco_annotation_path = f"demo/dataset/fasciola_ori/annotations/instances_{set_name}.json",
        coco_image_folder = f"demo/dataset/fasciola_ori/{set_name}",
        output_folder = "demo/dataset/fasciola_yolo",
        output_set_name = set_name
    )
    
'''YOLO --> COCO
This code uses the ultralytics/yolov5 format:
- {yolo_image_dir}
    - {image_1}
    - {image_2}
    - ...
- {yolo_label_dir}
    - {image_1}
    - {image_2}
    - ...
'''

for set_name in ["train", "test", "val"]:
    yolo_to_coco(
        yolo_image_dir = f"demo/dataset/fasciola_yolo/images/{set_name}",
        yolo_label_dir = f"demo/dataset/fasciola_yolo/labels/{set_name}",
        yolo_class_file = "demo/dataset/fasciola_yolo/classes.txt",
        coco_image_dir = f"demo/dataset/fasciola_coco/{set_name}",
        coco_annotation_path = f"demo/dataset/fasciola_coco/annotations/instances_{set_name}.json"
    )
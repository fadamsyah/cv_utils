from cv_utils.object_detection.dataset.converter import coco_to_yolo

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
'''

project_name = 'sel_telur'
for set_name in ["train", "test", "val"]:
    coco_to_yolo(coco_annotation_path = f"test/{project_name}/annotations/instances_{set_name}.json",
                 coco_image_folder = f"test/{project_name}/{set_name}",
                 output_folder = "test/dataset",
                 output_set_name = set_name)

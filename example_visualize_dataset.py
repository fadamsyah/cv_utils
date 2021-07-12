from cv_utils.object_detection.dataset.visualizer import visualize_yolo
from cv_utils.object_detection.dataset.visualizer import visualize_coco
   
visualize_yolo(
    image_path = 'demo/dataset/fasciola_yolo/images/test/49c17b90910b4a.jpg',
    label_path = 'demo/dataset/fasciola_yolo/labels/test/49c17b90910b4a.txt',
    yolo_class_file = 'demo/dataset/fasciola_yolo/classes.txt', show = True,
    save_labeled_path = None, max_image_to_screen_ratio = 0.8,
    seed = 10
)

visualize_coco(
    image_path = 'demo/dataset/fasciola_ori/test/222e90ca7a0444.jpg',
    annotation_path = 'demo/dataset/fasciola_ori/annotations/instances_test.json',
    show = False, save_labeled_path = 'demo/dataset/sample.jpg',
    max_image_to_screen_ratio = 0.8, seed = 42
)

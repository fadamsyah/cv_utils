from cv_utils.object_detection.dataset.visualizer import visualize_yolo
from cv_utils.object_detection.dataset.visualizer import visualize_coco
   
visualize_yolo('test/dataset/images/test/1a22388ab2a54e.jpg',
               'test/dataset/labels/test/1a22388ab2a54e.txt',
               'test/dataset/classes.txt', True, 'test/demo.jpg', 0.8)

visualize_coco('test/sel_telur/test/49c17b90910b4a.jpg',
               'test/sel_telur/annotations/instances_test.json',
               True, 'test/demo_coco.jpg', 0.8)
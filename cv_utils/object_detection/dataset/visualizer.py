import cv2
import json
import os
import pyautogui
import random

from copy import deepcopy

from .utils import read_json
    
def visualize_coco(image_path, annotation_path,
                   show = True, save_labeled_path = None,
                   max_image_to_screen_ratio = None, seed=42):
        
    # Read the annotation file
    coco_annotation = read_json(annotation_path)
        
    # Get the classes
    coco_class = [category['name']
                  for category in coco_annotation["categories"]]
    
    # Get a unique color for each class
    colors = get_class_color(coco_class, seed=seed)
    
    # Get the image_id of the image
    for image in coco_annotation["images"]:
        if image["file_name"] == os.path.basename(image_path):
            image_id = int(image["id"])
            break
    
    # Construct a universal-formatted bounding-boxes
    bbs = []
    for annotation in coco_annotation["annotations"]:
        if int(annotation["image_id"]) != image_id:
            continue
        
        # Get bounding-box coordinates
        xmin, ymin, w, h = annotation["bbox"]
        xmax = int(xmin + w)
        ymax = int(ymin + h)
        xmin = int(xmin)
        ymin = int(ymin)
        
        # In the COCO format, category_id starts from 1
        category_id = int(annotation["category_id"]) - 1
        
        # Get the corresponding color
        color = colors[category_id]
        bbs.append([xmin, ymin, xmax, ymax, color])
    
    # Get the image
    img = cv2.imread(image_path)
    
    visualize(img, bbs, show, save_labeled_path,
              max_image_to_screen_ratio)
    
def visualize_yolo(image_path, label_path, yolo_class_file,
                   show=True, save_labeled_path = None,
                   max_image_to_screen_ratio = None, seed=42):
    
    # Get dataset class from a .txt file
    with open(yolo_class_file, "r") as f:
        yolo_class = f.read().splitlines()
            
    # Get a unique color for each class
    colors = get_class_color(yolo_class, seed=seed)
    
    # Get the image
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    
    # Get bounding-boxes from the corresponding label
    with open(label_path, "r") as f:
        annotations = f.read().splitlines()
    
    # Construct a universal-formatted bounding-boxes
    bbs = []
    for bb in annotations:
        # YOLO: (category_id, xc, yc, w, h)
        # In the YOLO format, category_id starts from 0
        category_id, xc, yc, w, h = bb.split(' ')
        
        # xc, yc, w, h are normalized
        # So we need to unnormalize
        xmin = max(float(xc) - float(w)/2., 0.) * width
        ymin = max(float(yc) - float(h)/2., 0.) * height
        xmax = min(float(xc) + float(w)/2., 1.) * width
        ymax = min(float(yc) + float(h)/2., 1.) * height
        
        # Add the formatted bounding-box
        color = colors[int(category_id)]
        bbs.append([xmin, ymin, xmax, ymax, color])
    
    visualize(img, bbs, show, save_labeled_path,
              max_image_to_screen_ratio)

def get_class_color(classes, seed=42):
    random.seed(seed)
    colors = {i: (random.randint(0, 255),
                  random.randint(0, 255),
                  random.randint(0, 255))
              for i in range(len(classes)+1)}
    
    return colors

def visualize(img, bbs, show=True, save_labeled_path=None,
              max_image_to_screen_ratio=0.75):
    
    if save_labeled_path is not None:
        # Draw bounding-boxes
        img_c = draw_bbs(img, bbs)
        
        # Save the image
        cv2.imwrite(save_labeled_path, img_c)
    
    # Resize the image
    if max_image_to_screen_ratio is not None:
        img_c, bbs_c = resize_image(img, bbs, max_image_to_screen_ratio)
    else: img_c, bbs_c = img.copy(), deepcopy(bbs)
    
    # Draw bounding-boxes
    img_c = draw_bbs(img_c, bbs_c)
    
    if show:
        cv2.imshow("Sample", img_c)
        cv2.waitKey(0)

def draw_bbs(img, bbs):
    img_c = img.copy()
    for bb in bbs:
        xmin, ymin, xmax, ymax, color = bb
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        cv2.rectangle(img_c, (xmin, ymin), (xmax, ymax), color, 2)
        
    return img_c

def resize_image(img, bbs, max_image_to_screen_ratio):

    s_w, s_h = pyautogui.size() # Get screen width & height
    i_h, i_w = img.shape[:2] # Get image width & height
    
    # Get aspect ratio
    ratio = s_h / i_h * max_image_to_screen_ratio
    if ratio * i_w >= s_w:
        ratio = s_w / i_w * max_image_to_screen_ratio
    
    # Resize the image
    img_c = cv2.resize(img, (int(i_w*ratio), int(i_h*ratio)),
                       interpolation=cv2.INTER_CUBIC)
    
    # Update the bounding boxes
    bbs_c = []
    for bb in bbs:
        bbs_c.append([
            bb[0]*ratio, bb[1]*ratio, bb[2]*ratio,
            bb[3]*ratio, bb[-1]
        ])
    
    return img_c, bbs_c
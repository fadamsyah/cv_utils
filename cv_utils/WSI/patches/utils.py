import cv2
import numpy as np
import openslide
from skimage.filters import threshold_otsu

def get_thumbnail(slide, thumbnail_level):
    thumbnail_size = slide.level_dimensions[thumbnail_level]
    thumbnail = slide.get_thumbnail(thumbnail_size)
    thumbnail = np.array(thumbnail)
    
    return thumbnail

def get_hsv_otsu_threshold(img):
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    hthresh = threshold_otsu(h)
    sthresh = threshold_otsu(s)
    vthresh = threshold_otsu(v)
    
    return hsv_image, hthresh, sthresh, vthresh
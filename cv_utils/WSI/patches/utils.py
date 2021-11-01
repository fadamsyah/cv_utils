import cv2
import numpy as np
import openslide

from PIL import Image
from skimage.filters import threshold_otsu

from ..utils import pil_to_cv2

def get_slide_crop(slide, loc_crop, level, patch_size):
    crop_slide = slide.read_region(loc_crop, level, patch_size)
    crop_slide = pil_to_cv2(crop_slide)
    
    return crop_slide

def get_thumbnail(slide, inp, interpolation=Image.BICUBIC):
    if isinstance(inp, int):
        thumbnail_size = slide.level_dimensions[inp]
    elif isinstance(inp, (list, tuple)):
        thumbnail_size = inp
    
    for i in range(len(slide.level_dimensions)):
        current_slide_dim = slide.level_dimensions[i]
        if (current_slide_dim[0] < thumbnail_size[0]) and \
            (current_slide_dim[1] < thumbnail_size[1]):
            i -= 1; break
    
    thumbnail = slide.read_region((0,0), i, slide.level_dimensions[i])
    thumbnail.thumbnail(thumbnail_size, interpolation)
    thumbnail = pil_to_cv2(thumbnail)
    
    return thumbnail

def get_hsv_otsu_threshold(img):
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    hthresh = threshold_otsu(h)
    sthresh = threshold_otsu(s)
    vthresh = threshold_otsu(v)
    
    return hsv_image, hthresh, sthresh, vthresh

def get_size(size):
    if isinstance(size, (list, tuple)):
        size_x, size_y = size
    elif isinstance(size, int):
        size_x, size_y = size, size
    return size_x, size_y
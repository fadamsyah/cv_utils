import cv2
import numpy as np
import openslide
import os

from .functional import get_patches_coor
from .utils import get_thumbnail
from .utils import get_hsv_otsu_threshold

def generate_patches(slide, mask, level, patch_size, stride, save_dir,
                     thumbnail_level=6, drop_last=True, h_max=180,
                     s_max=255, v_min=70):
    """
    [BELUM SELESAI]
    """
    
    x_org_size, y_org_size = slide.level_dimensions[0]
    
    patches_coor = get_patches_coor(x_org_size, y_org_size, level,
                                    patch_size, stride, drop_last)
    
    thumbnail = get_thumbnail(slide, thumbnail_level)
    
    hsv_image, hthresh, sthresh, vthresh = get_hsv_otsu_threshold(thumbnail)
    
    hsv_min = np.array([hthresh, sthresh, v_min], np.uint8)
    hsv_max = np.array([h_max, s_max, vthresh], np.uint8)
    
    tissue_region_binary = cv2.inRange(hsv_image, hsv_min, hsv_max)
    
    print(hthresh, sthresh, vthresh)
    cv2.imwrite('trb.png', tissue_region_binary)
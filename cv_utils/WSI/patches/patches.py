import cv2
import numpy as np
import openslide
import os

from .functional import get_patches_coor

def generate_patches(slide, mask, level, patch_size, stride,
                     save_dir, drop_last=True):
    """
    [BELUM SELESAI]
    """
    
    x_org_size, y_org_size = slide.level_dimensions[0]
    
    patches_coor = get_patches_coor(x_org_size, y_org_size, level,
                                    patch_size, stride, drop_last)
    
    print(len(patches_coor))
    print(patches_coor[-1])
    print(slide.level_dimensions[level])
import albumentations as A
import cv2
import numpy as np
import openslide
import os

from .patches import get_crop_mask, get_loc_crop, is_foreground
from .hard_neg_samples import (
    generate_patches_from_coors,
    generate_hard_negative_coors,
    helper_read
    )
from .utils import get_slide_crop

def generate_hard_positive_samples(
    path_slide, path_mask, path_thumbnail_mask, path_thumbnail_heatmap,
    patch_size, inspection_size, stride, min_pct_tumor_area, min_mstd, max_threshold,
    level, save_dir, ext_patch='png', ext_mask='png', max_samples=1_000
    ):
    thumbnail_mask, thumbnail_heatmap = helper_read(path_thumbnail_mask, path_thumbnail_heatmap)
    
    prefix = os.path.split(path_slide)[1].split('.')[0]
    prefix = f"{prefix}"
    
    coors = generate_hard_positive_coors(thumbnail_mask, thumbnail_heatmap, max_threshold)
    
    slide = openslide.OpenSlide(path_slide)
    mask = openslide.OpenSlide(path_mask)
    generate_positive_patches_from_coors(slide, mask, level, coors, patch_size, inspection_size,
                                         stride, min_pct_tumor_area, min_mstd, max_samples, prefix,
                                         save_dir, ext_patch, ext_mask)

def generate_hard_positive_coors(mask, heatmap, max_threshold=0.5):    
    coors = generate_hard_negative_coors(1. - mask.copy(),
                                         1. - heatmap.copy(),
                                         1. - max_threshold)
    coors = [(1. - coor[0], coor[1], coor[2]) for coor in coors]
    
    return coors

def generate_positive_patches_from_coors(
    slide, mask, level, coors, patch_size, inspection_size, stride,
    min_pct_tumor_area, min_mstd, max_samples, prefix, save_dir,
    ext_patch='png', ext_mask='png'
    ):
    
    class Filter():
        def __init__(self):
            multiplier = pow(2, level)
            inspection_size_x, inspection_size_y = inspection_size
            inspection_size_x = inspection_size_x // multiplier
            inspection_size_y = inspection_size_y // multiplier
            self.centercrop = A.CenterCrop(inspection_size_y, inspection_size_x,
                                           always_apply=True)
            self.min_tumor_area = (inspection_size_x*inspection_size_y) * min_pct_tumor_area
            self.min_mstd = min_mstd
        
        def __call__(self, crop_slide, crop_mask):
            cond_1 = self.cond_mstd(crop_slide)
            cond_2 = self.cond_tumor_area(crop_mask)
            
            throw_away = cond_1 or cond_2
            
            return throw_away
        
        def cond_mstd(self, crop_slide):
            center_crop_slide = self.centercrop(image=crop_slide)['image']
            
            return not is_foreground(center_crop_slide, self.min_mstd)
        
        def cond_tumor_area(self, crop_mask):
            tumor_area = cv2.countNonZero(self.centercrop(image=crop_mask)['image'])
            
            return not tumor_area >= self.min_tumor_area
    
    generate_patches_from_coors(slide, mask, level, coors, patch_size, stride,
                                max_samples, prefix, save_dir, Filter(),
                                ext_patch, ext_mask)
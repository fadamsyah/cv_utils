import albumentations as A
import cv2
import numpy as np
import openslide
import os

from .patches import get_crop_mask, get_loc_crop
from .utils import get_slide_crop

def generate_hard_negative_samples(
    path_slide, path_mask, path_thumbnail_mask, path_thumbnail_heatmap,
    patch_size, inspection_size, stride, min_threshold, level, save_dir,
    ext_patch='png', ext_mask='png', max_samples=1_000
    ):
    thumbnail_mask, thumbnail_heatmap = helper_read(path_thumbnail_mask, path_thumbnail_heatmap)
    
    prefix = os.path.split(path_slide)[1].split('.')[0]
    prefix = f"{prefix}"
    
    coors = generate_hard_negative_coors(thumbnail_mask, thumbnail_heatmap, min_threshold)
    
    slide = openslide.OpenSlide(path_slide)
    mask = openslide.OpenSlide(path_mask)
    generate_negative_patches_from_coors(slide, mask, level, coors, patch_size, inspection_size, stride,
                                         max_samples, prefix, save_dir, ext_patch, ext_mask)

def generate_hard_negative_coors(mask, heatmap, min_threshold=0.5):
    
    fp = np.where(mask==1., 0., heatmap)
    
    x_axis = np.empty_like(fp, dtype=np.int32)
    y_axis = np.empty_like(fp, dtype=np.int32)
    
    for i in range(x_axis.shape[1]): x_axis[:,i]=i
    for j in range(y_axis.shape[0]): y_axis[j,:]=j
    
    fp = fp.ravel()[:, np.newaxis]
    x_axis = x_axis.ravel()[:, np.newaxis]
    y_axis = y_axis.ravel()[:, np.newaxis]
    
    res = np.concatenate((fp, x_axis, y_axis), axis=-1)
    
    res = list(filter(lambda var: var[0] >= min_threshold, res))
    res = sorted(res, reverse=True, key=lambda var:var[0])
    coors = [(coor[0], int(coor[1]), int(coor[2])) for coor in res]
    
    return coors

def generate_negative_patches_from_coors(
    slide, mask, level, coors, patch_size, inspection_size, stride,
    max_samples, prefix, save_dir, ext_patch='png', ext_mask='png'
    ):
    
    class Filter():
        def __init__(self):
            multiplier = pow(2, level)
            inspection_size_x, inspection_size_y = inspection_size
            inspection_size_x = inspection_size_x // multiplier
            inspection_size_y = inspection_size_y // multiplier
            self.centercrop = A.CenterCrop(inspection_size_y, inspection_size_x,
                                           always_apply=True)
        
        def __call__(self, crop_slide, crop_mask):
            return cv2.countNonZero(self.centercrop(image=crop_mask.copy())['image']) > 0
    
    generate_patches_from_coors(slide, mask, level, coors, patch_size, stride,
                                max_samples, prefix, save_dir, Filter(),
                                ext_patch, ext_mask)

def generate_patches_from_coors(
    slide, mask, level, coors, patch_size, stride, max_samples,
    prefix, save_dir, filter_object, ext_patch='png', ext_mask='png'
    ):
    save_tmp = os.path.join(save_dir, prefix)
    
    i = 0
    for coor in coors:
        loc_crop = get_loc_crop(coor[1:], patch_size, stride)
        crop_slide = get_slide_crop(slide, loc_crop, level, patch_size)
        crop_mask = get_crop_mask(mask, loc_crop, level, patch_size)
        
        if filter_object(crop_slide, crop_mask):
            continue
        
        filename = f"{save_tmp}_{loc_crop[0]}_{loc_crop[1]}"
        patch_name = f"{filename}_patch.{ext_patch}"
        crop_mask_name = f"{filename}_mask.{ext_mask}"
        
        if os.path.exists(patch_name): continue
        
        cv2.imwrite(patch_name, crop_slide)
        if ext_mask is not None:
            cv2.imwrite(crop_mask_name, crop_mask)
        
        i += 1
        if i >= max_samples: break

def helper_read(path_thumbnail_mask, path_thumbnail_heatmap):
    thumbnail_mask = cv2.imread(path_thumbnail_mask).astype(np.float32)
    thumbnail_heatmap = cv2.imread(path_thumbnail_heatmap).astype(np.float32)
    
    if thumbnail_mask.max() == 255: thumbnail_mask = thumbnail_mask / 255.
    if thumbnail_heatmap.max() > 1: thumbnail_heatmap = thumbnail_heatmap / 255.
    
    return thumbnail_mask[:,:,0], thumbnail_heatmap[:,:,0]
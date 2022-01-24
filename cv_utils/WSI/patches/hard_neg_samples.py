import albumentations as A
import cv2
import numpy as np
import openslide
import os
import random

from typing import List, Tuple, Union, Optional

from .patches import get_crop_mask, get_loc_crop
from .utils import get_slide_crop

def generate_hard_negative_samples(
    path_slide: str,
    path_mask: str,
    path_thumbnail_mask: str,
    path_thumbnail_heatmap: str,
    patch_size: Union[int, List[int], Tuple[int, int]],
    inspection_size: Union[List[int], Tuple[int, int]],
    stride: Union[int, List[int], Tuple[int, int]],
    min_threshold: float,
    level: int,
    save_dir: str,
    ext_patch: str = 'png',
    ext_mask: str = 'png',
    max_samples: int = 1_000,
    shuffle: bool = False,
    ) -> int:
    
    thumbnail_mask, thumbnail_heatmap = helper_read(path_thumbnail_mask, path_thumbnail_heatmap)
    
    prefix = os.path.split(path_slide)[1].split('.')[0]
    prefix = f"{prefix}"
    
    coors = generate_hard_negative_coors(thumbnail_mask, thumbnail_heatmap, min_threshold, shuffle)
    
    slide = openslide.OpenSlide(path_slide)
    mask = openslide.OpenSlide(path_mask)
    n_samples = generate_negative_patches_from_coors(slide, mask, level, coors, patch_size,
                                                     inspection_size, stride, max_samples,
                                                     prefix, save_dir, ext_patch, ext_mask)
    
    return n_samples

def generate_hard_negative_coors(
    mask: np.ndarray,
    heatmap: np.ndarray,
    min_threshold: float = 0.5,
    shuffle: bool = False,
    ) -> List[Union[List, Tuple[float, int, int]]]:
    
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
    
    if shuffle:
        coors = random.sample(coors, len(coors))
    
    return coors

def generate_negative_patches_from_coors(
    slide: openslide.OpenSlide,
    mask: openslide.OpenSlide,
    level: int,
    coors: List[Union[List, Tuple[float, int, int]]],
    patch_size: Union[int, List[int], Tuple[int, int]],
    inspection_size: Union[List[int], Tuple[int, int]],
    stride: Union[int, List[int], Tuple[int, int]],
    max_samples: int,
    prefix: str,
    save_dir: str,
    ext_patch: str = 'png',
    ext_mask: str = 'png',
    zero_pixel_in_outer: bool = True,
    ) -> int:
    
    class Filter():
        def __init__(self) -> None:
            multiplier = pow(2, level)
            inspection_size_x, inspection_size_y = inspection_size
            inspection_size_x = inspection_size_x // multiplier
            inspection_size_y = inspection_size_y // multiplier
            self.centercrop = A.CenterCrop(inspection_size_y, inspection_size_x,
                                           always_apply=True)
        
        def __call__(
            self,
            crop_slide: np.ndarray,
            crop_mask: np.ndarray,
            ) -> bool:
            
            if zero_pixel_in_outer==True:
                return cv2.countNonZero(crop_mask) > 0
            else:
                return cv2.countNonZero(self.centercrop(image=crop_mask.copy())['image']) > 0
    
    n_samples = generate_patches_from_coors(slide, mask, level, coors, patch_size, stride,
                                            max_samples, prefix, save_dir, Filter(),
                                            ext_patch, ext_mask)
    
    return n_samples

def generate_patches_from_coors(
    slide: openslide.OpenSlide,
    mask: openslide.OpenSlide,
    level: int,
    coors: List[Union[List, Tuple[float, int, int]]],
    patch_size: Union[int, List[int], Tuple[int, int]],
    stride: Union[int, List[int], Tuple[int, int]],
    max_samples: int,
    prefix: str,
    save_dir: str,
    filter_object,
    ext_patch: str = 'png',
    ext_mask: str = 'png',
    ) -> int:
    
    save_tmp = os.path.join(save_dir, prefix)
    
    i = 0
    for coor in coors:
        if i >= max_samples: break
        
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
    
    return i

def helper_read(
    path_thumbnail_mask: str,
    path_thumbnail_heatmap: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
    
    thumbnail_mask = cv2.imread(path_thumbnail_mask).astype(np.float32)
    thumbnail_heatmap = cv2.imread(path_thumbnail_heatmap).astype(np.float32)
    
    if thumbnail_mask.max() == 255: thumbnail_mask = thumbnail_mask / 255.
    if thumbnail_heatmap.max() > 1: thumbnail_heatmap = thumbnail_heatmap / 255.
    
    return thumbnail_mask[:,:,0], thumbnail_heatmap[:,:,0]
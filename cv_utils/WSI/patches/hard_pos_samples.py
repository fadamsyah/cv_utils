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
    path_slide: str,
    path_mask: str,
    path_thumbnail_mask: str,
    path_thumbnail_heatmap: str,
    patch_size: Union[int, List[int, int], Tuple[int, int]],
    inspection_size: Union[List[int, int], Tuple[int, int]],
    stride: Union[int, List[int, int], Tuple[int, int]],
    min_pct_tumor_area: float,
    min_mstd: float,
    max_threshold: float,
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
    
    coors = generate_hard_positive_coors(thumbnail_mask, thumbnail_heatmap, max_threshold, shuffle)
    
    slide = openslide.OpenSlide(path_slide)
    mask = openslide.OpenSlide(path_mask)
    n_samples = generate_positive_patches_from_coors(slide, mask, level, coors, patch_size,
                                                     inspection_size, stride, min_pct_tumor_area,
                                                     min_mstd, max_samples, prefix, save_dir,
                                                     ext_patch, ext_mask)
    
    return n_samples

def generate_hard_positive_coors(
    mask: np.ndarray,
    heatmap: np.ndarray,
    max_threshold: float = 0.5,
    shuffle: bool = False,
    ) -> List[Union[List[float, int, int], Tuple[float, int, int]]]:
    
    coors = generate_hard_negative_coors(1. - mask,
                                         1. - heatmap,
                                         1. - max_threshold,
                                         shuffle)
    coors = [(1. - coor[0], coor[1], coor[2]) for coor in coors]
    
    return coors

def generate_positive_patches_from_coors(
    slide: openslide.OpenSlide,
    mask: openslide.OpenSlide,
    level: int,
    coors: List[Union[List[float, int, int], Tuple[float, int, int]]],
    patch_size: Union[int, List[int, int], Tuple[int, int]],
    inspection_size: Union[List[int, int], Tuple[int, int]],
    stride: Union[int, List[int, int], Tuple[int, int]],
    min_pct_tumor_area: float,
    min_mstd: float,
    max_samples: int,
    prefix: str,
    save_dir: str,
    ext_patch: str = 'png',
    ext_mask: str = 'png',
    ) -> int:
    
    class Filter():
        def __init__(self) -> None:
            multiplier = pow(2, level)
            inspection_size_x, inspection_size_y = inspection_size
            inspection_size_x = inspection_size_x // multiplier
            inspection_size_y = inspection_size_y // multiplier
            self.centercrop = A.CenterCrop(inspection_size_y, inspection_size_x,
                                           always_apply=True)
            self.min_tumor_area = (inspection_size_x*inspection_size_y) * min_pct_tumor_area
            self.min_mstd = min_mstd
        
        def __call__(
            self,
            crop_slide: np.ndarray,
            crop_mask: np.ndarray,
            ) -> bool:
            
            cond_1 = self.cond_mstd(crop_slide)
            cond_2 = self.cond_tumor_area(crop_mask)
            
            throw_away = cond_1 or cond_2
            
            return throw_away
        
        def cond_mstd(self, crop_slide: np.ndarray) -> bool:
            center_crop_slide = self.centercrop(image=crop_slide)['image']
            
            return not is_foreground(center_crop_slide, self.min_mstd)
        
        def cond_tumor_area(self, crop_mask: np.ndarray) -> bool:
            tumor_area = cv2.countNonZero(self.centercrop(image=crop_mask)['image'])
            
            return not tumor_area >= self.min_tumor_area
    
    n_samples = generate_patches_from_coors(slide, mask, level, coors, patch_size,
                                            stride, max_samples, prefix, save_dir,
                                            Filter(), ext_patch, ext_mask)
    
    return n_samples
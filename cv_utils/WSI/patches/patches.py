"""
References:
- https://github.com/3dimaging/DeepLearningCamelyon
"""

import albumentations as A
import cv2
import numpy as np
import openslide
import os

from pathlib import Path
from skimage.filters import threshold_otsu
from tqdm import tqdm
from typing import List, Tuple, Union, Optional

from .utils import get_size
from .utils import get_thumbnail
from .utils import get_slide_crop

from ...utils import create_and_overwrite_dir

def calculate_tumor_patches(
    path_slide: str,
    path_mask: str,
    level: int,
    patch_size: Union[int, List[int], Tuple[int, int]],
    stride: Union[int, List[int], Tuple[int, int]],
    inspection_size: Union[int, List[int], Tuple[int, int]],
    min_mstd: float = 5.,
    min_pct_tumor_area: float = 0.05,
    max_tumor_patches: Optional[int] = None,
    automate_scaling: bool = True,
    debug: bool = True,
    ) -> int:
    
    multiplier = pow(2, level)
    
    # Read slide & mask
    slide = openslide.OpenSlide(path_slide)
    mask = openslide.OpenSlide(path_mask)
    
    # Get stride, patch_size, and inspection_size for each x, y coordinates
    stride_x, stride_y = get_size(stride)
    patch_size_x, patch_size_y = get_size(patch_size)
    inspection_size_x, inspection_size_y = get_size(inspection_size)
    
    # Get thumbnail size
    x_org_size, y_org_size = slide.level_dimensions[0]
    x_tmb_size = int(x_org_size / stride_x)
    y_tmb_size = int(y_org_size / stride_y)
    
    # Get a slide thumbnail, a tissue binary map, and coordinates containing tissues
    _, tissue_coordinates, _ = get_tissue_coordinates(slide, x_tmb_size, y_tmb_size)
    
    # Get a tumor thumbnail, a tumor binary map, and coordinates containing tumors
    _, tumor_coordinates = get_tumor_coordinates(mask, x_tmb_size, y_tmb_size)
    
    if debug:
        print(f"Path slide: {path_slide}")
        print(f"Path mask: {path_mask}")
        
        print(f"\nStride: ({stride_x}, {stride_y})")
        print(f"Patch size: ({patch_size_x}, {patch_size_y})")
        print(f"Inspection size: ({inspection_size_x}, {inspection_size_y})")
        
        print(f'\nRegion Distribution:')
        print(f"The number of segmented tissue regions: {len(tissue_coordinates)}")
        print(f"The number of tumor regions: {len(tumor_coordinates)}")
    
    # For filtering tissue and tumor regions
    if automate_scaling == True:
        inspection_size_x = inspection_size_x // multiplier
        inspection_size_y = inspection_size_y // multiplier
    centercrop = A.CenterCrop(inspection_size_y, inspection_size_x, always_apply=True)
    min_tumor_area = (inspection_size_x*inspection_size_y) * min_pct_tumor_area
    
    # Patches of tumor region
    if debug: print("\nCalculate tumor patches ...")
    n_tumor_patches = 0
    iterations = tumor_coordinates
    if debug: iterations = tqdm(iterations)
    for coor in iterations:
        loc_crop = get_loc_crop(coor, patch_size, stride)
        crop_slide = get_slide_crop(slide, loc_crop, level, (patch_size_x, patch_size_y), automate_scaling)
        
        if min_mstd is not None:
            center_crop_slide = centercrop(image=crop_slide)['image']
            if not is_foreground(center_crop_slide, min_mstd): continue
        
        # Check whether this is a tumor
        crop_mask = get_crop_mask(mask, loc_crop, level, (patch_size_x, patch_size_y), automate_scaling)
        tumor_area = cv2.countNonZero(centercrop(image=crop_mask)['image'])
        category = 'tumor' if tumor_area >= min_tumor_area else 'normal'
        
        # If it is a tumor, then
        if category == 'tumor':
            n_tumor_patches += 1
        
        if max_tumor_patches:
            if n_tumor_patches >= max_tumor_patches: break
    
    if debug: print(f"\nNumber of tumor patches: {n_tumor_patches}")
    
    return n_tumor_patches

def generate_training_patches(
    path_slide: str,
    path_mask: str,
    level: int,
    patch_size: Union[int, List[int], Tuple[int, int]],
    stride: Union[int, List[int], Tuple[int, int]],
    inspection_size: Union[int, List[int], Tuple[int, int]],
    save_dir: str,
    normal_tumor_ratio: float = 1.0,
    min_mstd: float = 5.,
    min_pct_tumor_area: float = 0.05,
    max_pct_tumor_area_in_normal_patch: float = 0.,
    max_tumor_patches: Optional[int] = None,
    max_normal_backgrounds: Optional[int] = None,
    ext_patch: str = 'tif',
    ext_mask: str = 'tif',
    overwrite: bool = False,
    normal_check_all_pixels: bool = True,
    automate_scaling = True,
    debug: bool = True,
    ) -> None:
    
    multiplier = pow(2, level)
    
    # Create or overwrite dirname
    for _, dirname in save_dir.items():
        if overwrite:
            create_and_overwrite_dir(dirname)
        else:
            Path(dirname).mkdir(parents=True, exist_ok=True)
    
    # Read slide & mask
    slide = openslide.OpenSlide(path_slide)
    mask = openslide.OpenSlide(path_mask)
    
    # Get stride, patch_size, and inspection_size for each x, y coordinates
    stride_x, stride_y = get_size(stride)
    patch_size_x, patch_size_y = get_size(patch_size)
    inspection_size_x, inspection_size_y = get_size(inspection_size)
    
    # Get thumbnail size
    x_org_size, y_org_size = slide.level_dimensions[0]
    x_tmb_size = int(x_org_size / stride_x)
    y_tmb_size = int(y_org_size / stride_y)
    
    # Get a slide thumbnail, a tissue binary map, and coordinates containing tissues
    _, tissue_coordinates, _ = get_tissue_coordinates(slide, x_tmb_size, y_tmb_size)
    
    # Get a tumor thumbnail, a tumor binary map, and coordinates containing tumors
    _, tumor_coordinates = get_tumor_coordinates(mask, x_tmb_size, y_tmb_size)
    
    if debug:
        print(f"Path slide: {path_slide}")
        print(f"Path mask: {path_mask}")
        
        print(f"\nStride: ({stride_x}, {stride_y})")
        print(f"Patch size: ({patch_size_x}, {patch_size_y})")
        print(f"Inspection size: ({inspection_size_x}, {inspection_size_y})")
        
        print(f'\nRegion Distribution:')
        print(f"The number of segmented tissue regions: {len(tissue_coordinates)}")
        print(f"The number of tumor regions: {len(tumor_coordinates)}")
    
    # For filtering tissue and tumor regions
    if automate_scaling == True:
        inspection_size_x = inspection_size_x // multiplier
        inspection_size_y = inspection_size_y // multiplier
    centercrop = A.CenterCrop(inspection_size_y, inspection_size_x, always_apply=True)
    min_tumor_area = (inspection_size_x*inspection_size_y) * min_pct_tumor_area
    
    # Patches of tumor region
    if debug: print("\nGenerate tumor patches ...")
    n_tumor_patches = 0
    iterations = tumor_coordinates
    if debug: iterations = tqdm(iterations)
    for coor in iterations:
        loc_crop = get_loc_crop(coor, patch_size, stride)
        crop_slide = get_slide_crop(slide, loc_crop, level, (patch_size_x, patch_size_y), automate_scaling)
        
        if min_mstd is not None:
            center_crop_slide = centercrop(image=crop_slide)['image']
            if not is_foreground(center_crop_slide, min_mstd): continue
        
        # Check whether this is a tumor
        crop_mask = get_crop_mask(mask, loc_crop, level, (patch_size_x, patch_size_y), automate_scaling)
        tumor_area = cv2.countNonZero(centercrop(image=crop_mask)['image'])
        category = 'tumor' if tumor_area >= min_tumor_area else 'normal'
        
        # If it is a tumor, then
        if category == 'tumor':
            n_tumor_patches += 1
            filename = os.path.split(path_slide)[1].split('.')[0]
            filename = f"{filename}_{loc_crop[0]}_{loc_crop[1]}"
            
            cv2.imwrite(os.path.join(save_dir[category], f"{level}_{filename}_patch.{ext_patch}"), crop_slide)
            if ext_mask is not None: cv2.imwrite(os.path.join(save_dir[category], f"{level}_{filename}_mask.{ext_mask}"), crop_mask)
        
        if max_tumor_patches:
            if n_tumor_patches >= max_tumor_patches: break
    
    # Patches of normal region
    if debug: print("\nGenerate normal patches ...")
    max_tumor_area = (inspection_size_x*inspection_size_y) * max_pct_tumor_area_in_normal_patch
    n_normal_patches = 0
    n_normal_background = 0
    iterations = tissue_coordinates
    if debug: iterations = tqdm(iterations)
    for coor in iterations:
        loc_crop = get_loc_crop(coor, patch_size, stride)
        crop_slide = get_slide_crop(slide, loc_crop, level, (patch_size_x, patch_size_y), automate_scaling)
        
        # Check whether this is a normal patch or not
        crop_mask = get_crop_mask(mask, loc_crop, level, (patch_size_x, patch_size_y), automate_scaling)
        if normal_check_all_pixels==True:
            tumor_area = cv2.countNonZero(crop_mask)
        else:
            tumor_area = cv2.countNonZero(centercrop(image=crop_mask)['image'])
        category = 'normal' if tumor_area <= max_tumor_area else 'unnormal'
        
        if category != 'normal': continue
        
        if min_mstd is not None:
            center_crop_slide = centercrop(image=crop_slide)['image']
            if not is_foreground(center_crop_slide, min_mstd):
                if max_normal_backgrounds is not None:
                    if n_normal_background <= max_normal_backgrounds:
                        filename = os.path.split(path_slide)[1].split('.')[0]
                        filename = f"{filename}_{loc_crop[0]}_{loc_crop[1]}"
                        cv2.imwrite(os.path.join(save_dir[category], f"{level}_{filename}_patch.{ext_patch}"), crop_slide)
                        if ext_mask is not None: cv2.imwrite(os.path.join(save_dir[category], f"{level}_{filename}_mask.{ext_mask}"), crop_mask)
                        n_normal_background += 1
                continue
        
        # If it is a normal patch, then
        if category == 'normal':
            n_normal_patches += 1
            filename = os.path.split(path_slide)[1].split('.')[0]
            filename = f"{filename}_{loc_crop[0]}_{loc_crop[1]}"
            
            cv2.imwrite(os.path.join(save_dir[category], f"{level}_{filename}_patch.{ext_patch}"), crop_slide)
            if ext_mask is not None: cv2.imwrite(os.path.join(save_dir[category], f"{level}_{filename}_mask.{ext_mask}"), crop_mask)
        
        if n_normal_patches >= (n_tumor_patches*normal_tumor_ratio): break
    
    if debug:
        print('\nNumber of classes:')
        print(f"tumor : {n_tumor_patches}")
        print(f"normal: {n_normal_patches}")

def get_tissue_coordinates(
    slide: Union[str, openslide.OpenSlide],
    x_tmb_size: int,
    y_tmb_size: int,
    ksize: Optional[int] = 3,
    kernel_size: Union[Tuple[int, int], List[int]] = (2,2),
    iterations: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    if isinstance(slide, str):
        slide = openslide.OpenSlide(slide)
    
    tissue_thumbnail = get_thumbnail(slide, (x_tmb_size, y_tmb_size))
    tissue_binary_map = process_thumbnail_binary_map(tissue_thumbnail.copy(), 'slide',
                                                     ksize, kernel_size, iterations)
    tissue_coordinates = get_positive_coordinates(tissue_binary_map)
    np.random.shuffle(tissue_coordinates)
    
    return tissue_thumbnail, tissue_coordinates, tissue_binary_map

def get_tumor_coordinates(
    mask: Union[str, openslide.OpenSlide],
    x_tmb_size: int,
    y_tmb_size: int,
    ksize: Optional[int] = None,
    kernel_size: Union[Tuple[int, int], List[int]] = (3,3),
    iterations: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
    
    if isinstance(mask, str):
        mask = openslide.OpenSlide(mask)
    
    tumor_thumbnail = get_thumbnail(mask, (x_tmb_size, y_tmb_size))
    tumor_binary_map = process_thumbnail_binary_map(tumor_thumbnail, 'mask',
                                                    ksize, kernel_size, iterations)
    tumor_coordinates = get_positive_coordinates(tumor_binary_map)
    np.random.shuffle(tumor_coordinates)
    
    return tumor_thumbnail, tumor_coordinates

def get_positive_coordinates(
    binary_map: np.ndarray,
    ) -> np.ndarray:
    
    # Remember that the OpenCV library uses (H x W x C) format, whereas
    # OpenSlide uses (W x H x C) format.
    list_y, list_x = np.where(binary_map == 255)
    list_x = list_x[:, np.newaxis]
    list_y = list_y[:, np.newaxis]
    coordinates = np.concatenate((list_x, list_y), axis=-1)
    
    return coordinates

def get_loc_crop(
    coordinate: Union[List[int], Tuple[int, int]],
    patch_size: Union[int, List[int], Tuple[int, int]],
    stride: Union[int, List[int], Tuple[int, int]],
    ) -> Tuple[int, int]:
    
    # Get stride and patch_size for each x, y coordinates
    stride_x, stride_y = get_size(stride)
    patch_size_x, patch_size_y = get_size(patch_size)
    
    i, j = coordinate
    
    loc_crop = (i*stride_x - (patch_size_x - stride_x)//2,
                j*stride_y - (patch_size_y - stride_y)//2)
    
    return loc_crop

def get_crop_mask(
    mask: openslide.OpenSlide,
    loc_crop: Union[List[int], Tuple[int, int]],
    level: int,
    patch_size: Union[List[int], Tuple[int, int]],
    automate_scaling: bool = True,
    ) -> np.ndarray:
    
    crop_mask = get_slide_crop(mask, loc_crop, level, patch_size, automate_scaling)
    crop_mask = cv2.cvtColor(crop_mask, cv2.COLOR_BGR2GRAY)
    crop_mask = np.where(crop_mask != 0, 255, 0).astype(np.uint8)
    
    return crop_mask

def process_thumbnail_binary_map(
    thumbnail: np.ndarray,
    slide_type: str,
    ksize: Optional[int] = None,
    kernel_size: Optional[Union[Tuple[int, int], List[int]]] = None,
    iterations: Optional[int] = None,
    ) -> np.ndarray:
    
    grey = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2GRAY)
    
    if slide_type.lower() == 'slide':
        thresh = threshold_otsu(grey)
        binary_map = np.where(grey < thresh, 255, 0).astype(np.uint8)
    elif slide_type.lower() == 'mask':
        binary_map = np.where(grey >= 127, 255, 0).astype(np.uint8)
    
    if ksize:
        binary_map = cv2.medianBlur(binary_map, ksize=ksize)
    if kernel_size and iterations:
        binary_map = cv2.dilate(binary_map, np.ones(kernel_size, np.uint8),
                                iterations=iterations)
    
    return binary_map

def is_foreground(
    patch: np.ndarray,
    threshold_mstd: float = 5.,
    ) -> bool:
    
    # Reference/Source:
    # https://github.com/longluu/DL-CancerDetection-CAMELYON16/blob/master/CancerDetection_preprocessing_1level.ipynb
    mstd = np.mean(np.std(patch, axis=-1))
    
    return mstd >= threshold_mstd
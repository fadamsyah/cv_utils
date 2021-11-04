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

from .functional import get_patches_coor
from .utils import get_size
from .utils import get_thumbnail
from .utils import get_hsv_otsu_threshold
from .utils import get_slide_crop

from ...utils import create_and_overwrite_dir

def generate_training_patches(path_slide, path_mask, level, patch_size, stride,
                              inspection_size, save_dir, drop_last=True, h_max=180,
                              s_max=255, v_min=70, min_pct_tissue_area=0.1,
                              min_pct_tumor_area=0.05, ext='tif', overwrite=False):
    """
    - Baru bisa untuk level 0 saja
    """
    
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
    tissue_thumbnail = get_thumbnail(slide, (x_tmb_size, y_tmb_size))
    tissue_binary_map = process_thumbnail_binary_map(tissue_thumbnail.copy(),
                                                     'slide', 3, (2,2), 1)
    tissue_coordinates = get_positive_coordinates(tissue_binary_map)
    np.random.shuffle(tissue_coordinates)
    
    # Get a HSV threshold
    hsv_image, hthresh, sthresh, vthresh = get_hsv_otsu_threshold(tissue_thumbnail)
    hsv_min = np.array([hthresh, sthresh, v_min], np.uint8)
    hsv_max = np.array([h_max, s_max, vthresh], np.uint8)
    
    # Get a tumor thumbnail, a tumor binary map, and coordinates containing tumors
    tumor_thumbnail = get_thumbnail(mask, (x_tmb_size, y_tmb_size))
    tumor_binary_map = process_thumbnail_binary_map(tumor_thumbnail, 'mask',
                                                    3, (3,3), 1)
    tumor_coordinates = get_positive_coordinates(tumor_binary_map)
    
    print(f"Path slide: {path_slide}")
    print(f"Path mask: {path_mask}")
    
    print(f"\nStride: ({stride_x}, {stride_y})")
    print(f"Patch size: ({patch_size_x}, {patch_size_y})")
    print(f"Inspection size: ({inspection_size_x}, {inspection_size_y})")
    
    print(f'\nRegion Distribution:')
    print(f"The number of segmented tissue regions: {len(tissue_coordinates)}")
    print(f"The number of tumor regions: {len(tumor_coordinates)}")
    
    # For filtering tissue and tumor regions
    centercrop = A.CenterCrop(inspection_size_y, inspection_size_x, always_apply=True)
    min_tissue_area = (inspection_size_x*inspection_size_y) * min_pct_tissue_area \
        if min_pct_tissue_area is not None else None
    min_tumor_area = (inspection_size_x*inspection_size_y) * min_pct_tumor_area
    
    # Patches of tumor region
    print("\nGenerate tumor patches ...")
    n_tumor_patches = 0
    for coor in tqdm(tumor_coordinates):
        loc_crop = get_loc_crop(coor, patch_size, stride)
        crop_slide = get_slide_crop(slide, loc_crop, level, (patch_size_x, patch_size_y))
        
        if min_tissue_area is not None:
            crop_tissue_binary = centercrop(image=crop_slide)['image']
            crop_tissue_binary = cv2.inRange(cv2.cvtColor(crop_tissue_binary, cv2.COLOR_BGR2HSV),
                                             hsv_min, hsv_max)
            if cv2.countNonZero(crop_tissue_binary) < min_tissue_area: continue
        
        # Check whether this is a tumor
        crop_mask = get_crop_mask(mask, loc_crop, level, (patch_size_x, patch_size_y))
        tumor_area = cv2.countNonZero(centercrop(image=crop_mask)['image'])
        category = 'tumor' if tumor_area >= min_tumor_area else 'normal'
        
        # If it is a tumor, then
        if category == 'tumor':
            n_tumor_patches += 1
            filename = os.path.split(path_slide)[1].split('.')[0]
            filename = f"{filename}_{loc_crop[0]}_{loc_crop[1]}"
            
            cv2.imwrite(os.path.join(save_dir[category], f"{filename}_patch.{ext}"), crop_slide)
            cv2.imwrite(os.path.join(save_dir[category], f"{filename}_mask.{ext}"), crop_mask)
    
    # Patches of normal region
    print("\nGenerate normal patches ...")
    n_normal_patches = 0
    for coor in tqdm(tissue_coordinates):
        loc_crop = get_loc_crop(coor, patch_size, stride)
        crop_slide = get_slide_crop(slide, loc_crop, level, (patch_size_x, patch_size_y))
        
        if min_tissue_area is not None:
            crop_tissue_binary = centercrop(image=crop_slide)['image']
            crop_tissue_binary = cv2.inRange(cv2.cvtColor(crop_tissue_binary, cv2.COLOR_BGR2HSV),
                                             hsv_min, hsv_max)
            if cv2.countNonZero(crop_tissue_binary) < min_tissue_area: continue
        
        # Check whether this is a tumor
        crop_mask = get_crop_mask(mask, loc_crop, level, (patch_size_x, patch_size_y))
        tumor_area = cv2.countNonZero(centercrop(image=crop_mask)['image'])
        category = 'tumor' if tumor_area >= min_tumor_area else 'normal'
        
        # If it is not a tumor, then
        if category == 'normal':
            n_normal_patches += 1
            filename = os.path.split(path_slide)[1].split('.')[0]
            filename = f"{filename}_{loc_crop[0]}_{loc_crop[1]}"
            
            cv2.imwrite(os.path.join(save_dir[category], f"{filename}_patch.{ext}"), crop_slide)
            cv2.imwrite(os.path.join(save_dir[category], f"{filename}_mask.{ext}"), crop_mask)
        
        if n_normal_patches >= n_tumor_patches: break

    print('\nNumber of classes:')
    print(f"tumor : {n_tumor_patches}")
    print(f"normal: {n_normal_patches}")

def get_positive_coordinates(binary_map):
    # Remember that the OpenCV library uses (H x W x C) format, whereas
    # OpenSlide uses (W x H x C) format.
    list_y, list_x = np.where(binary_map == 255)
    list_x = list_x[:, np.newaxis]
    list_y = list_y[:, np.newaxis]
    coordinates = np.concatenate((list_x, list_y), axis=-1)
    
    return coordinates

def get_loc_crop(coordinate, patch_size, stride):
    # Get stride and patch_size for each x, y coordinates
    stride_x, stride_y = get_size(stride)
    patch_size_x, patch_size_y = get_size(patch_size)
    
    i, j = coordinate
    
    loc_crop = (i*stride_x - (patch_size_x - stride_x)//2,
                j*stride_y - (patch_size_y - stride_y)//2)
    
    return loc_crop

def get_crop_mask(mask, loc_crop, level, patch_size):
    crop_mask = get_slide_crop(mask, loc_crop, level, patch_size)
    crop_mask = cv2.cvtColor(crop_mask, cv2.COLOR_BGR2GRAY)
    crop_mask = np.where(crop_mask != 0, 255, 0).astype(np.uint8)
    
    return crop_mask

def process_thumbnail_binary_map(thumbnail, slide_type, ksize=3, kernel_size=(2,2), iterations=1):
    grey = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2GRAY)
    
    if slide_type.lower() == 'slide':
        thresh = threshold_otsu(grey)
        binary_map = np.where(grey < thresh, 255, 0).astype(np.uint8)
    elif slide_type.lower() == 'mask':
        binary_map = np.where(grey >= 127, 255, 0).astype(np.uint8)
    
    binary_map = cv2.medianBlur(binary_map, ksize=ksize)
    binary_map = cv2.dilate(binary_map, np.ones(kernel_size, np.uint8),
                            iterations=iterations)
    
    return binary_map
import albumentations as A
import cv2
import numpy as np
import openslide
import os

from skimage.filters import threshold_otsu
from tqdm import tqdm

from .functional import get_patches_coor
from .utils import get_size
from .utils import get_thumbnail
from .utils import get_hsv_otsu_threshold

from ... import create_and_overwrite_dir

def generate_patches(path_slide, path_mask, level, patch_size, stride,
                     save_dir, drop_last=True, h_max=180, s_max=255,
                     v_min=70, min_pct_tissue_area=0.05,
                     min_pct_tumor_area=0.1, ext='tif'):
    """
    - Baru bisa untuk level 0 saja
    """
    
    # Create and overwrite dirname
    for _, dirname in save_dir.items():
        create_and_overwrite_dir(dirname)
    
    # Read slide & mask
    slide = openslide.OpenSlide(path_slide)
    mask = openslide.OpenSlide(path_mask)
    
    # Get stride and patch_size for each x, y coordinates
    stride_x, stride_y = get_size(stride)
    patch_size_x, patch_size_y = get_size(patch_size)
    
    # Get thumbnail size
    x_org_size, y_org_size = slide.level_dimensions[0]
    x_tmb_size = int(x_org_size / stride_x)
    y_tmb_size = int(y_org_size / stride_y)
    
    # Get a thumbnail of slide
    thumbnail = get_thumbnail(slide, (x_tmb_size, y_tmb_size))
    
    # Get HSV threshold
    hsv_image, hthresh, sthresh, vthresh = get_hsv_otsu_threshold(thumbnail)
    hsv_min = np.array([hthresh, sthresh, v_min], np.uint8)
    hsv_max = np.array([h_max, s_max, vthresh], np.uint8)
    
    # Get tissue_binary_map
    grey = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2GRAY)
    thresh = threshold_otsu(grey)
    tissue_binary_map = np.where(grey < thresh, 255, 0).astype(np.uint8)
    
    # Get tissue coordinates
    list_i, list_j = np.where(tissue_binary_map == 255)
    list_i = list_i[:, np.newaxis]
    list_j = list_j[:, np.newaxis]
    coordinates = np.concatenate((list_i, list_j), axis=-1)
    
    centercrop = A.CenterCrop(stride_y, stride_x, always_apply=True)
    min_tissue_area = (stride_x*stride_y) * min_pct_tissue_area \
        if min_pct_tissue_area is not None else None
    min_tumor_area = (stride_x*stride_y) * min_pct_tumor_area
    
    # Process crops
    for coor in tqdm(coordinates):
        i, j = coor        
        loc_crop = (i*stride_x - (patch_size_x - stride_x)//2,
                    j*stride_y - (patch_size_y - stride_y)//2)
        
        crop_slide = np.array(slide.read_region(loc_crop, level, (patch_size_x, patch_size_y))).astype(np.uint8)
        
        if min_tissue_area is not None:
            crop_tissue_binary = centercrop(image=crop_slide)['image']
            crop_tissue_binary = cv2.inRange(cv2.cvtColor(crop_tissue_binary, cv2.COLOR_BGR2HSV), hsv_min, hsv_max)
            if cv2.countNonZero(crop_tissue_binary) < min_tissue_area: continue
        
        crop_mask = np.array(mask.read_region(loc_crop, level, (patch_size_x, patch_size_y))).astype(np.uint8)
        crop_mask = cv2.cvtColor(crop_mask, cv2.COLOR_BGR2GRAY)
        crop_mask = np.where(crop_mask != 0, 255, 0).astype(np.uint8)
        
        # Check whether this is a tumor
        tumor_area = cv2.countNonZero(centercrop(image=crop_mask)['image'])
        typ = 'tumor' if tumor_area >= min_tumor_area else 'normal'
        
        # Save the patch
        filename = os.path.split(path_slide)[1].split('.')[0]
        filename = f"{filename}_{loc_crop[0]}_{loc_crop[1]}"
        
        cv2.imwrite(os.path.join(save_dir[typ], f"{filename}_patch.{ext}"), crop_slide)
        cv2.imwrite(os.path.join(save_dir[typ], f"{filename}_mask.{ext}"), crop_mask)
        

# THE RUNNING TIME IS EXTREMELY SLOW
# def generate_patches(slide, mask, level, patch_size, stride, inspection_size, min_pct_tissue_area,
#                      save_dir, thumbnail_level=6, drop_last=True, h_max=180,
#                      s_max=255, v_min=70):
#     """
#     [BELUM SELESAI]
#     """
    
#     if isinstance(inspection_size, (list, dict, tuple)):
#         inspection_size_x, inspection_size_y = inspection_size
#     elif isinstance(inspection_size, int):
#         inspection_size_x, inspection_size_y = inspection_size, inspection_size
    
#     x_org_size, y_org_size = slide.level_dimensions[0]
    
#     patches_coor = get_patches_coor(x_org_size, y_org_size, level,
#                                     patch_size, stride, drop_last)
    
#     thumbnail = get_thumbnail(slide, thumbnail_level)
    
#     hsv_image, hthresh, sthresh, vthresh = get_hsv_otsu_threshold(thumbnail)
    
#     hsv_min = np.array([hthresh, sthresh, v_min], np.uint8)
#     hsv_max = np.array([h_max, s_max, vthresh], np.uint8)
    
#     min_tissue_area = inspection_size_x * inspection_size_y * min_pct_tissue_area
#     for coor in tqdm(patches_coor):
#         crop_slide = np.array(slide.read_region(coor['location'], coor['level'], coor['size']))
        
#         tissue_binary = cv2.inRange(cv2.cvtColor(crop_slide, cv2.COLOR_BGR2HSV), hsv_min, hsv_max)
        
#         if cv2.countNonZero(tissue_binary) >= min_tissue_area:
#             crop_mask = np.array(mask.read_region(coor['location'], coor['level'], coor['size']))
#             crop_mask = cv2.cvtColor(crop_mask, cv2.COLOR_BGR2GRAY)
#             crop_mask = np.where(crop_mask > 0, 255, 0).astype(np.uint8)
            
#             location = coor['location']
#             filename = f"patches/{location[0]}_{location[1]}"
#             cv2.imwrite(f"{filename}_0.png", crop_slide)
#             cv2.imwrite(f"{filename}_1.png", crop_mask)
"""
References:
- https://github.com/3dimaging/DeepLearningCamelyon
"""

import cv2
import math
import multiresolutionimageinterface as mir
import numpy as np
import openslide
import os
import pandas as pd

from .utils import xml_to_df

def viz_with_xml(path_wsi, path_xml, level):
    reader = mir.MultiResolutionImageReader()
    mr_image = reader.open(path_wsi)
    
    wsi_dim_x, wsi_dim_y = mr_image.getDimensions()
    dims = mr_image.getLevelDimensions(level)
    
    scale_x = dims[0] / wsi_dim_x
    scale_y = dims[1] / wsi_dim_y
    annotations = xml_to_df(path_xml, scale_x, scale_y)
    
    final_list = []
    for num in annotations['Name']:
        if num not in final_list:
            final_list.append(num)
    
    coxy = [[] for x in range(len(final_list))]
    
    i = 0
    for n in final_list:
        newx = annotations[annotations['Name']==n]['X']
        newy = annotations[annotations['Name']==n]['Y']
        print(n)
        print(newx, newy)
        newxy = list(zip(newx, newy))
        coxy[i] = np.array(newxy, dtype=np.int32)
        i=i+1
    
    tile = mr_image.getUCharPatch(0, 0, dims[0], dims[1], level)
    vis = cv2.drawContours(tile, coxy, -1, (0, 255, 0), 10)
    
    return vis

def viz_with_mask(path_wsi, path_mask, level):
    slide = openslide.open_slide(path_wsi)
    mask = openslide.open_slide(path_mask)
    
    rgb_image = slide.read_region((0, 0), level, slide.level_dimensions[level])
    rgb_mask = truth.read_region((0, 0), level, slide.level_dimensions[level])
    
    grey = np.array(rgb_mask.convert('L'))
    rgb_imagenew = np.array(rgb_image)
    
    _, contours, _ = cv2.findContours(grey, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vis = cv2.drawContours(rgb_imagenew, contours, -1, (0, 0, 255), 5)
    
    return vis
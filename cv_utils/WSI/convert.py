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

from typing import List

def xml_to_mask(
    path_wsi: str,
    path_xml: str,
    path_mask: str,
    label_map: dict = {'_0': 255, '_1': 255, '_2': 0},
    conversion_order: List[str] = ['_0', '_1', '_2'],
    ) -> None:
    
    reader = mir.MultiResolutionImageReader()
    mr_image = reader.open(path_wsi)
    
    annotation_list = mir.AnnotationList()
    
    xml_repository = mir.XmlRepository(annotation_list)
    xml_repository.setSource(path_xml)
    xml_repository.load()
    
    annotation_mask = mir.AnnotationToMask()
    annotation_mask.convert(annotation_list, path_mask, mr_image.getDimensions(),
                            mr_image.getSpacing(), label_map, conversion_order)
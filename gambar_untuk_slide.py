import cv2
import numpy as np
import openslide

from cv_utils.WSI import get_thumbnail, get_slide_crop
from cv_utils.WSI.patches import process_thumbnail_binary_map

PATH_SLIDE = 'WSI/slide/tumor/tumor_084.tif'
PATH_MASK = 'WSI/mask/tumor/tumor_084.tif'
LEVEL = 5
BB_SIZE = (20,20)
BB_THICKNESS = 3
BB_COLOR = (0, 0, 0)

min_tissue_area = 0.1 * BB_SIZE[0] * BB_SIZE[1]

slide = openslide.OpenSlide(PATH_SLIDE)
mask = openslide.OpenSlide(PATH_MASK)
x_tmb_size, y_tmb_size = slide.level_dimensions[LEVEL]

thumbnail = get_thumbnail(slide, (x_tmb_size, y_tmb_size))
tissue_map = process_thumbnail_binary_map(thumbnail, 'slide', 3, (2,2), 1)
thumbnail_tumor = get_thumbnail(mask, (x_tmb_size, y_tmb_size))
tumor_map = process_thumbnail_binary_map(thumbnail_tumor, 'mask', 3, (2,2), 1)

cv2.imwrite('img-1.png', thumbnail)
cv2.imwrite('img-2.png', tissue_map)

list_area_loc = []

i = 0
while True:
    j = 0
    while True:
        crop = tissue_map[i:i+BB_SIZE[0], j:j+BB_SIZE[1]]
        
        nt = cv2.countNonZero(crop)
        if nt > min_tissue_area:
            cv2.rectangle(thumbnail, (j,i), (j+BB_SIZE[1], i+BB_SIZE[0]),
                            BB_COLOR, BB_THICKNESS)
            list_area_loc.append([nt, [j,i]])
        
        j += BB_SIZE[1]
        if j + BB_SIZE[1] >= thumbnail.shape[1]:
            break
    i += BB_SIZE[0]
    if i + BB_SIZE[0] >= thumbnail.shape[0]:
        break

cv2.imwrite('img-3.png', thumbnail)
cv2.imwrite('img-4.png', tumor_map)

for area in sorted(list_area_loc)[-5:]:
    x, y = area[1]
    crop = get_slide_crop(slide, (x*BB_SIZE[1], y*BB_SIZE[0]), 0, (256,256))
    
    cv2.imwrite(f'crop_{x}_{y}.png', crop)
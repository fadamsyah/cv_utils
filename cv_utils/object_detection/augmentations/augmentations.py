import albumentations as A
import math
import random
from albumentations import DualTransform
from albumentations.augmentations.crops import functional as F

class CustomCropNearBBox(DualTransform):
    # ONLY FOR 1 BBOX
    def __init__(self, always_apply=False, p=1.0):
        super(CustomCropNearBBox, self).__init__(always_apply, p)
    
    def get_params_dependent_on_targets(self, params):
        img = params['image']
        height, width = img.shape[:2]
        
        bbox = params['bboxes'][0][:4]
        bbox = A.denormalize_bbox(bbox, height, width)
        bbox = list(map(math.ceil, bbox))
        
        x_min = max(bbox[0] - random.randint(0, bbox[0]), 0)
        y_min = max(bbox[1] - random.randint(0, bbox[1]), 0)
        x_max = min(bbox[2] + random.randint(0, width - bbox[2]), width)
        y_max = min(bbox[3] + random.randint(0, height - bbox[3]), height)
        
        return {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}
    
    def apply(self, img, x_min=0, x_max=0, y_min=0, y_max=0, **params):
        return F.clamping_crop(img, x_min, y_min, x_max, y_max)
    
    def apply_to_bbox(self, bbox, x_min=0, x_max=0, y_min=0, y_max=0, **params):
        return F.bbox_crop(bbox, x_min, y_min, x_max, y_max, **params)
    
    @property
    def targets_as_params(self):
        return ["image", "bboxes"]
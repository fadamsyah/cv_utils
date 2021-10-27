import math

from .utils import get_size

def get_patches_coor(x_org_size, y_org_size, level, patch_size, stride,
                     drop_last=True):
    
    stride_x, stride_y = get_size(stride)
    
    if isinstance(patch_size, (list, dict, tuple)):
        patch_size_x, patch_size_y = patch_size
    elif isinstance(patch_size, int):
        patch_size_x, patch_size_y = patch_size, patch_size
    
    x_size = math.ceil(x_org_size / (level + 1))
    y_size = math.ceil(y_org_size / (level + 1))
        
    res = []
    x_coor, y_coor = 0, 0
    while True:
        while True:
            res.append({'location': (x_coor, y_coor), 'level': level,
                        'size': (patch_size_x, patch_size_y)})
            x_coor += stride_x
            if (x_coor + patch_size_x) > x_size:
                x_coor = 0; break
        y_coor += stride_y
        if (y_coor + patch_size_y) > y_size:
            y_coor = 0; break
    
    return res
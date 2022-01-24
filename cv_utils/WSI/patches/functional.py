import math

from typing import List, Tuple, Union

from .utils import get_size

def get_patches_coor(
    x_org_size: int,
    y_org_size: int,
    level: int,
    patch_size: Union[int, List[int], Tuple[int, int]],
    stride: Union[int, List[int], Tuple[int, int]],
    drop_last: bool = True,
    ) -> List[dict]:
    
    stride_x, stride_y = get_size(stride)
    
    if isinstance(patch_size, (list, tuple)):
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
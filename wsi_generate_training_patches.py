import argparse

from cv_utils import WSI
from cv_utils import boolean_string

"""
Example:
    - python wsi_generate_training_patches.py -w WSI/slide/tumor/tumor_019.tif \
        -m WSI/mask/tumor/tumor_019.tif --ext png --overwrite True
"""

parser = argparse.ArgumentParser(description='Visualize Annotation')
parser.add_argument('-w', '--wsi', type=str, default=None, help='WSI path')
parser.add_argument('-m', '--mask', type=str, default=None, help='mask path')
parser.add_argument('-e', '--ext', type=str, default='png', help='patches extension')
parser.add_argument('-o', '--overwrite', type=boolean_string, default=False,
                    help='(True or False) overwrite the save directories')
args = parser.parse_args()

level = 0
patch_size = (300, 300)
stride = (64, 64)
inspection_size = (128, 128)
min_mstd = 5.
min_pct_tumor_area = 0.5
save_dir = {'tumor': 'WSI/patches/tumor',
            'normal': 'WSI/patches/normal'}

WSI.generate_training_patches(
    args.wsi, args.mask, level, patch_size, stride, inspection_size, save_dir,
    min_mstd=min_mstd, min_pct_tumor_area=min_pct_tumor_area, ext=args.ext,
    overwrite=args.overwrite
)
import argparse
import cv2
import openslide

from cv_utils import WSI
from cv_utils import boolean_string

"""
Example
    - python wsi_viz.py -l 4 -w WSI/slide/tumor/tumor_019.tif -x WSI/xml/tumor/tumor_019.xml -m WSI/mask/tumor/tumor_019.tif \
        -s vis.png --show True
"""

parser = argparse.ArgumentParser(description='Visualize Annotation')
parser.add_argument('-w', '--wsi', type=str, default=None, help='WSI path')
parser.add_argument('-x', '--xml', type=str, default=None, help='XML path')
parser.add_argument('-m', '--mask', type=str, default=None, help='mask path')
parser.add_argument('-l', '--level', type=int, default=4, help='pyramid level')
parser.add_argument('-s', '--save_path', type=str, default=None, help='save path')
parser.add_argument('--show', type=boolean_string, default=False, help='(True or False) show the visualization')
args = parser.parse_args()

slide = openslide.OpenSlide(args.wsi)
print(f'Number of pyramid levels: {len(slide.level_dimensions)}')
print(slide.level_dimensions)

if args.xml:
    img = WSI.viz_with_xml(args.wsi, args.xml, args.level, args.show)

if args.mask:
    img = WSI.viz_with_mask(args.wsi, args.mask, args.level, args.show)

if args.save_path:
    cv2.imwrite(args.save_path, img)
import argparse

from cv_utils import WSI

"""
Example:
    - python wsi_xml_to_mask.py -w WSI/slide/tumor/tumor_019.tif \
        -x WSI/xml/tumor/tumor_019.xml -m WSI/mask/tumor/tumor_019.tif
"""

parser = argparse.ArgumentParser(description='Convert ASAP XML annotation to ')
parser.add_argument('-w', '--wsi', type=str, default=None, help='WSI path')
parser.add_argument('-x', '--xml', type=str, default=None, help='XML path')
parser.add_argument('-m', '--mask', type=str, default=None, help='mask path')
args = parser.parse_args()

WSI.xml_to_mask(args.wsi, args.xml, args.mask)
from cv_utils import WSI

PATH_WSI = 'WSI/slide/tumor/tumor_075.tif'
PATH_XML = 'WSI/xml/tumor/tumor_075.xml'
PATH_MASK = 'WSI/mask/tumor/tumor_075.tif'
LEVEL = 4

# Visualize using an annotation file (ASAP)
WSI.viz_with_xml(PATH_WSI, PATH_XML, LEVEL, True)

# Visualize using a mask file
WSI.viz_with_mask(PATH_WSI, PATH_MASK, LEVEL, True)
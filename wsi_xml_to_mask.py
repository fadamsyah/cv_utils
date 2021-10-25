from cv_utils import WSI

PATH_WSI = 'WSI/slide/tumor/tumor_075.tif'
PATH_XML = 'WSI/xml/tumor/tumor_075.xml'
PATH_MASK = 'WSI/mask/tumor/tumor_075.tif'

WSI.xml_to_mask(PATH_WSI, PATH_XML, PATH_MASK)
from .patches import (
    calculate_tumor_patches,
    get_patches_coor, get_tissue_coordinates,
    generate_hard_positive_samples, generate_hard_negative_samples,
    generate_training_patches,
    get_thumbnail, get_slide_crop
    )
from .utils import pil_to_cv2, xml_to_df
from .viz import viz_with_xml, viz_with_mask

try: from .convert import xml_to_mask
except:
    print("The multiresolutionimageinterface package is not installed.")
    print("Please refer to the ASAP repo if you want to install the package.")
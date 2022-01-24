"""
References:
- https://github.com/3dimaging/DeepLearningCamelyon
"""

import cv2
import math
import numpy as np
import pandas as pd
import PIL
import xml
import xml.etree.ElementTree as et

from typing import Union

def xml_to_df(
    inp: Union[str, et.ElementTree],
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    ) -> pd.DataFrame:
    
    if isinstance(inp, str):
        parseXML = et.parse(inp)
    elif isinstance(inp, et.ElementTree):
        parseXML = inp
    
    root = parseXML.getroot()
    dfcols = ['Name', 'Order', 'X', 'Y']
    df_xml = pd.DataFrame(columns=dfcols)
    for child in root.iter('Annotation'):
        for coordinate in child.iter('Coordinate'):
            Name = child.attrib.get('Name')
            Order = coordinate.attrib.get('Order')
            X_coord = float(coordinate.attrib.get('X'))
            X_coord = X_coord * scale_x
            Y_coord = float(coordinate.attrib.get('Y'))
            Y_coord = Y_coord * scale_y
            df_xml = df_xml.append(pd.Series([Name, Order, X_coord, Y_coord],
                                             index = dfcols), ignore_index=True)
            df_xml = pd.DataFrame(df_xml)
    
    return df_xml

def pil_to_cv2(
    img: PIL.Image.Image,
    ) -> np.ndarray:
    
    img = img.convert("RGB")
    img = np.asarray(img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    return img
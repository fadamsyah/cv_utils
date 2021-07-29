import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import time

from PIL import Image

from cv_utils.object_detection.dataset.visualizer import streamlit_visualize_yolo
from cv_utils.object_detection.dataset.visualizer import streamlit_visualize_coco

SERVICES = [
    "Object detection: Visualize bounding-boxes",
    "Object detection: Analyze dataset",
    "Object detection: Convert dataset format",
    "Object detection: Split dataset"
]

def ds_od_vis():
    def _data_uploader(img_type, label_type):
        left_column, right_column = st.beta_columns(2)
        img_path = left_column.file_uploader('Select an image', type=img_type)
        label_path = right_column.file_uploader('Select a label', type=label_type)
        return img_path, label_path
        
    def _coco():
        img, annotation = _data_uploader(["png", "jpg", "jpeg"], ["json"])
        if None not in [img, annotation]:
            # Convert to cv2 format
            img = np.array(Image.open(img).convert('RGB'))
            img = img[:, :, ::-1].copy()
            
            st.write(json.loads(annotation.read()))
            
    def _yolo():
        img, labels = _data_uploader(["png", "jpg", "jpeg"], ["txt", "xml"])
        yolo_class = st.file_uploader("Select a YOLO class file", type=["txt", "csv", "json"])
        if None not in [img, labels, yolo_class]:
            # Convert to cv2 format
            img = np.array(Image.open(img).convert('RGB'))
            img = img[:, :, ::-1].copy()
            
            # Convert to line of string, as usual
            labels = [line.decode("utf-8")  for line in labels]
            yolo_class = [line.decode("utf-8") for line in yolo_class]
                        
            # Create a matplotlib figure and turn-off its axes
            fig = plt.figure()
            plt.axis("off")            
            
            # Create a spinner for better experience
            st.spinner()
            with st.spinner(text="Visualizing the bounding-boxes. Please wait ..."):
                # Call the visualization function
                image = streamlit_visualize_yolo(img, labels, yolo_class)[:, :, ::-1]
                time.sleep(1.)
                st.success('Success')
            
            # Plot the image
            plt.imshow(image)
            st.pyplot(fig)
        
    def _pascal():
        pass
    
    dataset_format = ["coco", "yolo", "pascal"]
    vis_services = {
        dataset_format[0]: _coco,
        dataset_format[1]: _yolo,
        dataset_format[2]: _pascal
    }
    
    option = st.selectbox(
        "Select a dataset format",
        ["<select>"] + dataset_format
    )
    if option in dataset_format:
        vis_services[option]()


DICT_SERVICES = {
    SERVICES[0]: ds_od_vis
}

def main():
    st.header("Computer Vision Utilities")
    st.text("This repository is made to facilitate anyone working on computer vision.")
    st.text("We provide handy tools that can be utilized for your project.")
    
    option = st.selectbox(
        "Which service do you want to use?",
        ["<select>"] + SERVICES
    )

    if option in SERVICES:
        DICT_SERVICES[option]()
    
if __name__ == "__main__":
    main()
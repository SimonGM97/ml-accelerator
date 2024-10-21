#!/usr/bin/env python3
# from ml_accelerator.streamlit.inference_page import build_inference_page
from ml_accelerator.streamlit.model_registry_page import build_model_registry_page
import streamlit as st
from PIL import Image
import os


# streamlit run scripts/web_app/web_app.py
if __name__ == '__main__':
    # Set Page Config
    st.set_page_config(layout="wide")

    # Define first row
    col0, col1, col2 = st.columns([3, 3, 3])

    # Show Ed Machina Logo
    col1.image(Image.open(os.path.join("resources", "images", "ml_accelerator_logo.png")), use_column_width=True)
    
    # Blank space
    st.write("#")
    st.write("#")
    st.write("#")

    # Sidebar
    sidebar = st.sidebar.selectbox('Choose Page:', ['New Inferences', 'Model Registry'])

    if sidebar == 'New Inferences':
        # Build Inference Page
        # build_inference_page()
        pass

    if sidebar == 'Model Registry':
        # Build Model Registry Page
        build_model_registry_page()
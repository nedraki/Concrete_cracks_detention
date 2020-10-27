from PIL import Image, ImageOps
import numpy as np
from img_classification import classification
import time
import streamlit as st

#st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("TESTING")
my_placeholder = st.empty()
my_placeholder.text("Upload an image to evaluate its classification: ")

upload_placeholder = st.empty()
uploaded_file = upload_placeholder.file_uploader(" ", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    upload_placeholder.empty()
    my_placeholder.image(image, caption='Uploaded file.', width=320)
    time.sleep(0.2)

    with st.spinner('Wait for it...'):
       
        my_bar = st.progress(0)
        
        for percent_complete in range(100):
            time.sleep(0.1)
            my_bar.progress(percent_complete + 1)
        time.sleep(0.1)

    label = classification(image, 'trained_models/model_fine_tuned_crack_detection.h5')
    if label == 0:
        st.success('LABEL O')
        
        st.balloons()
    else:
        st.success("LABEL 1")
        
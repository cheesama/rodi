from pororo import Pororo
from spacy import displacy
from PIL import Image

import streamlit as st
import numpy as np
import random

# Set page title
st.title("Rapid pOroro Demo Inferencer - Only OCR")

# Load ocr model
@st.cache(allow_output_mutation=True)
def load_ocr_model():
    with st.spinner("Loading OCR model..."):
        ocr_model = Pororo(task="ocr", lang="ko")

        return ocr_model

if __name__ == "__main__":
    ## optical character recognition
    st.subheader("Optical Character Recognition")
    ocr_model = load_ocr_model()
    uploaded_file = st.file_uploader(
        "Upload Image file", type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file))
        st.json(ocr_model(image, detail=True))


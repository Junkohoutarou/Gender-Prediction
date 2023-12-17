import streamlit as st
import pickle as pkl
import numpy as np
from PIL import Image

class_list = {'0': 'Female', '1': 'Male'}

# Load CSS
with open("styles.css") as f:
    custom_css = f.read()
st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)

# Load model and encoder
input_ec = open('ec_vinames.pkl', 'rb')
encoder = pkl.load(input_ec)

input_md = open('lrc_vinames.pkl', 'rb')
model = pkl.load(input_md)

st.title("Gender Prediction Based on Vietnamese Full Names")

# Load image
image = Image.open("vinames.png")
st.image(image)

st.header('Write a Vietnamese full name')
txt = st.text_area('', '')

if txt != '':
    if st.button('Predict'):
        feature_vector = encoder.transform([txt])
        label = str((model.predict(feature_vector))[0])

        st.header('Result')
        st.text(class_list[label])

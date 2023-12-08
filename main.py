import streamlit as st
import pickle as pkl
import numpy as np
from PIL import Image

class_list = {'0': 'Female', '1': 'Male'}
st.title("Gender Prediction Based on Vietnamese Names ")

#image = Image.open('ten.png')
#st.image(image)

input_ec = open('ec_vinames.pkl', 'rb')
encoder = pkl.load(input_ec)

input_md = open('lrc_vinames.pkl', 'rb')
model = pkl.load(input_md)

st.header('Write a feedback')
txt = st.text_area('', '')

if txt != '':
  if st.button('Predict'):
    feature_vector = encoder.transform([txt])
    label = str((model.predict(feature_vector))[0])

    st.header('Result')
    st.text(class_list[label])


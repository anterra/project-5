import keras
from gensim.models.keyedvectors import KeyedVectors
import keras.models
from tensorflow.keras.models import Model, Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.utils import np_utils
import gensim
from tensorflow.keras.utils import plot_model
from tensorflow import keras
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import sklearn
import tensorflow as tf
KERAS_BACKEND = tf


# forward_model = load_model("forward_model.h5")

st.markdown("hello", unsafe_allow_html=True)
st.title("Vinyasa Krama: Deep Learning Generated Yoga")
st.write("Yoga Application")
peak_pose = st.selectbox("Pick a peak pose", ["pigeon", "bird of paradise"])
st.button("Generate a class")

import streamlit as st
from helper.data_util import DataUtil
from helper.extra_utils import ASSETS
import pandas as pd
import numpy as np
import os


dataUtil = DataUtil()

st.title("hello")
st.sidebar.image(os.path.join(ASSETS, 'logo.png'))
option = st.sidebar.radio(label="option", options=['a', 'b', 'c'], index=1)

if option == 'a':
    st.write(pd.DataFrame(np.random.random((10, 5))))
elif option == 'b':

    filename = st.text_input('Enter a file path:')
    try:
        with open(os.path.join(ASSETS, filename, '.txt')) as input_name:
            st.text(input_name.read())
    except FileNotFoundError:
        st.error('File not found.')

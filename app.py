import streamlit as st
from helper.data_util import DataUtil
from helper.extra_utils import ASSETS
from helper.churn_analysis import ChurnAnalysis
from helper.topic_modeling import IntentUtils
from helper.inference_utils import InferenceUtils
import pandas as pd
import numpy as np
import os


st.title("Real time Intent Recognition with Churn Analysis")

# Todo: Resize this image or find a better one

st.sidebar.image(os.path.join(ASSETS, 'logo.png'))
option = st.sidebar.radio(
    label="option",
    options=[
        'Tutorial: How to use it?',
        'Churn Analysis',
        'Intent Recognition',
        'Inference'
        'c'],
    index=0
)

if option == 'Tutorial: How to use it?':
    st.markdown((
        '''
        How to use damn thing!
        '''
    ), unsafe_allow_html=True)

elif option == 'Churn Analysis':
    churnAnalysis = ChurnAnalysis(data_folder='./data/raw_data')
    churnAnalysis.run(save=True, output_path='./data/output_data')

    churn_options = st.radio(label='Mode', options=['EDA', 'Modeling'], index=0)

    if st.button('Go'):
        churnAnalysis.run()

elif option == 'Intent Recognition':
    intentUtils = IntentUtils()
    # try:
    #     with open(os.path.join(ASSETS, filename, '.txt')) as input_name:
    #         st.text(input_name.read())
    # except FileNotFoundError:
    #     st.error('File not found.')

elif option == 'Inference':
    try:
        inferenceUtils = InferenceUtils()
    except ValueError as e:
        st.markdown("!!!Run Churn and Intent First!!!")



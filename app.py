from helper.churn_analysis_v2 import ChurnAnalysis
# from helper.topic_modeling import IntentUtils
# from helper.inference_utils import InferenceUtils
import os
import pickle

import streamlit as st

from helper.extra_utils import ASSETS, markup_text

# Title on every page
st.title("Improving Customer Experience")

# Navigational image
st.sidebar.image(os.path.join(ASSETS, 'logo.png'))

# Navigational options
option = st.sidebar.radio(
    label="Modules",
    options=[
        'Tutorial: How to use it?',
        'Churn Analysis',
        'Topic Modeling',
        'Inference'
    ],
    index=0
)

# option: 'Tutorial: How to use it?'
if option == 'Tutorial: How to use it?':
    description = markup_text.get('homepage', 'ERROR')
    st.markdown(description, unsafe_allow_html=True)

# option: 'Churn Module'
elif option == 'Churn Analysis':
    churnAnalysis = ChurnAnalysis(
        input_data_file_name='data.csv',
        data_input_path='./data/raw_data',
        data_output_path='./data/output',
        model_path='./data/model'
    )

    churn_module_options = st.sidebar.radio(
        label="What to do?",
        options=[
            'Exploratory Data Analysis',
            'Train Models',
            'Explainability',
        ],
        index=0
    )

    if churn_module_options == 'Exploratory Data Analysis':
        st.markdown("<b>Raw Data</b>", unsafe_allow_html=True)
        st.write(churnAnalysis.raw_data.head(4))
        st.markdown("<b>The columns of the data are</b>", unsafe_allow_html=True)
        st.write(churnAnalysis.show_col_name())

        figures = churnAnalysis.eda()

        st.write("DESCRIPTION")  # Todo: Add description here
        st.plotly_chart(figures)

        # st.write("DESCRIPTION")  # Todo: Add description here
        # st.plotly_chart(figures[1])
        #
        # st.write("DESCRIPTION")  # Todo: Add description here
        # st.plotly_chart(figures[2])
        #
        # st.write("DESCRIPTION")  # Todo: Add description here
        # st.plotly_chart(figures[3])

    elif churn_module_options == 'Train Models':
        model_options = st.radio(label='Select a model',
                                 options=[
                                     'Logistic Regression',
                                     'Random Forest',
                                     # 'Support Vector Machines'
                                 ], index=0)

        if st.button("Train"):
            reports = churnAnalysis.run(model_name=model_options,
                                        apply_reduction=True)
            train_report = reports['train_report']
            test_report = reports['train_report']
            st.write(train_report)
            st.write(test_report)
        else:
            X_test = pickle.load(open(os.path.join(churnAnalysis.data_output_path, 'x_test.pkl'), 'rb'))
            y_test = pickle.load(open(os.path.join(churnAnalysis.data_output_path, 'y_test.pkl'), 'rb'))

            test_report = churnAnalysis.predict(model_options, X_test, y_test)
            st.write(test_report)

        pass
    elif churn_module_options == 'Explainability':
        st.write("PLACEHOLDER")
        pass

elif option == 'Intent Recognition':
    st.write("PLACEHOLDER")
    # intentUtils = IntentUtils()
    # # try:
    # #     with open(os.path.join(ASSETS, filename, '.txt')) as input_name:
    # #         st.text(input_name.read())
    # # except FileNotFoundError:
    # #     st.error('File not found.')

elif option == 'Inference':
    st.write("PLACEHOLDER")
    # try:
    #     inferenceUtils = InferenceUtils()
    # except ValueError as e:
    #     st.markdown("!!!Run Churn and Intent First!!!")

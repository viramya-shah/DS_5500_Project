import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from churn_helper.churn_analysis import ChurnAnalysis
from churn_helper.explain import Explain
from churn_helper.extra_utils import ASSETS, markup_text
from churn_helper.streamlit_shap_plot_utils import set_block_container_style

from complaint_helper.sentiment_classifier import SentimentClassifier
from complaint_helper.topic_modeling import TopicModeling

st.set_option('deprecation.showPyplotGlobalUse', False)
set_block_container_style()

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

        st.markdown("<b>Overview</b>", unsafe_allow_html=True)  # Todo: Add description here
        st.plotly_chart(figures)

    elif churn_module_options == 'Train Models':
        model_options = st.radio(label='Select a model',
                                 options=[
                                     'Logistic Regression',
                                     'Random Forest',
                                     # 'Support Vector Machines'
                                 ], index=0)
        data_columns = churnAnalysis.feature_columns

        if st.button("Train"):
            st.text("Note: This might take a few minutes")
            reports = churnAnalysis.run(model_name=model_options,
                                        apply_reduction=False)
            train_report = reports['train_report']
            test_report = reports['test_report']

            X_test = pickle.load(open(os.path.join(churnAnalysis.data_output_path, 'x_test.pkl'), 'rb'))

            if model_options == 'Logistic Regression':
                model = pickle.load(open("./data/model/Logistic Regression/Logistic Regression.pkl",
                                         'rb')).best_estimator_
                coefficients = model['logistic'].coef_[0]

                plt.figure(figsize=(16, 5))
                ax = plt.gca()
                ax.set_xticklabels(X_test.columns, rotation=45)
                plt.bar(X_test.columns, list(coefficients))
                plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='both', alpha=0.7)
                plt.title("Coefficients for features")
                plt.ylabel("Coefficients")
                st.pyplot(plt.gcf())
                plt.clf()
            elif model_options == 'Random Forest':
                pass

            print(train_report)
            st.write("The classification report for the training set is as follows")
            report_train = pd.DataFrame(train_report)
            report_train.columns = ['Not Churn', 'Churn', 'accuracy', 'macro avg', 'weighted avg']
            st.write(report_train)

            st.write("The classification report for the test set is as follows")
            report_test = pd.DataFrame(test_report)
            report_test.columns = ['Not Churn', 'Churn', 'accuracy', 'macro avg', 'weighted avg']
            st.write(report_test)

        else:
            X_test = pickle.load(open(os.path.join(churnAnalysis.data_output_path, 'x_test.pkl'), 'rb'))
            y_test = pickle.load(open(os.path.join(churnAnalysis.data_output_path, 'y_test.pkl'), 'rb'))

            if model_options == 'Logistic Regression':
                model = pickle.load(open("./data/model/Logistic Regression/Logistic Regression.pkl",
                                         'rb')).best_estimator_
                coefficients = model['logistic'].coef_[0]

                plt.figure(figsize=(16, 5))
                ax = plt.gca()
                ax.set_xticklabels(X_test.columns, rotation=45)
                plt.bar(X_test.columns, list(coefficients))
                plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='both', alpha=0.7)
                plt.title("Coefficients for features")
                plt.ylabel("Coefficients")
                st.pyplot(plt.gcf())
                plt.clf()
            elif model_options == 'Random Forest':
                pass

            st.write("The classification report for the testing set is as follows")
            test_report = churnAnalysis.predict(model_options, X_test, y_test)
            report_test = pd.DataFrame(test_report)
            report_test.columns = ['Not Churn', 'Churn', 'accuracy', 'macro avg', 'weighted avg']
            st.write(report_test)

    elif churn_module_options == 'Explainability':
        model_options = st.radio(label='Select a model',
                                 options=[
                                     'Logistic Regression',
                                     'Random Forest',
                                     # 'Support Vector Machines'
                                 ], index=0)
        if st.button('Explain'):
            explain = Explain(model_name=model_options,
                              model_path='./data/model',
                              data_path='./data/output',
                              only_test=True)

            st.text("Note: This might take a few minutes")
            explain.run()

elif option == 'Topic Modeling':
    topic_model_options = st.sidebar.radio(
        label="What to do?",
        options=[
            'Exploratory Data Analysis',
            'Severity',
            'Extract Topics',
        ],
        index=0
    )
    if topic_model_options == 'Exploratory Data Analysis':
        sentimentClassifier = SentimentClassifier(data_path="./data/raw_data/",
                                                  file_name="sentiment_complaints.csv")
        st.plotly_chart(sentimentClassifier.eda())

    elif topic_model_options == 'Severity':
        text = st.text_input("Enter the text", "")

        if st.button("Go"):
            sentimentClassifier = SentimentClassifier(data_path="./data/raw_data/",
                                                      file_name="sentiment_complaints.csv")
            labels, info = sentimentClassifier.predict(text)
            st.markdown(f"<i><u>Severity</i></u>: <b>{labels}</b>",
                        unsafe_allow_html=True)

            st.markdown(markup_text.get("sentiments_extraction", "ERROR"))
            st.markdown(f"<i><u>Escalate here, if needed</u></i>: {info}", unsafe_allow_html=True)

    elif topic_model_options == 'Extract Topics':
        topicModeling = TopicModeling()
        text = st.text_input("Enter the text", "")

        extract_options = st.sidebar.radio(
            label="What to do?",
            options=[
                'Latent Dirichlet Allocation',  # Todo: change to full names
                'Non-Matrix Factorization',
            ],
            index=0
        )
        if extract_options == 'Latent Dirichlet Allocation':
            vectorizer = pickle.load(open("output/topic_modeling/tf_vectorizer.pkl", 'rb'))
            model = pickle.load(open("output/topic_modeling/lda.pkl", 'rb'))
        elif extract_options == 'Non-Matrix Factorization':
            vectorizer = pickle.load(open("output/topic_modeling/tfidf_vectorizer.pkl", 'rb'))
            model = pickle.load(open("output/topic_modeling/nmf.pkl", 'rb'))
        else:
            vectorizer = None
            model = None

        if st.button("Go"):
            topic, info = topicModeling.apply_predict_topic(text=text,
                                                            vectorizer=vectorizer,
                                                            lda_model=model)

            st.markdown(f"<i><u>Topic</i></u>: <b>{topic}</b>",
                        unsafe_allow_html=True)

            st.markdown(markup_text.get("topic_extraction", "ERROR"))
            st.markdown(f"<i><u>Escalate here, if needed</u></i>: {info}", unsafe_allow_html=True)

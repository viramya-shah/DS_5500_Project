import os
import pickle

import shap
import streamlit as st
import matplotlib.pyplot as plt
from churn_helper.streamlit_shap_plot_utils import show_plot, rf_summary_plot


class Explain:
    def __init__(self,
                 model_name: str = 'Logistic Regression',
                 model_path: str = './data/models',
                 data_path: str = './data/output',
                 only_test: bool = False,
                 ):
        """
        This class is responsible for all the functionality that is provided by the explainability module
        :param model_name: Input name for the model
        :param model_path: Input path for model dump
        :param data_path: Input path for the data
        :param only_test: Flag to explain only the test results
        """
        self.model_name = model_name
        self.model_path = model_path
        self.data_path = data_path
        self.only_test = only_test

        self.model = pickle.load(open(os.path.join(self.model_path,
                                                   model_name,
                                                   f'{model_name}.pkl'),
                                      'rb'))

        if not only_test:
            self.x_train = pickle.load(open(os.path.join(self.data_path,
                                                         'x_train.pkl'),
                                            'rb'))

            self.y_train = pickle.load(open(os.path.join(self.data_path,
                                                         'y_train.pkl'),
                                            'rb'))
        else:
            self.x_train = None
            self.y_train = None

        self.x_test = pickle.load(open(os.path.join(self.data_path,
                                                    'x_test.pkl'),
                                       'rb'))

        self.y_test = pickle.load(open(os.path.join(self.data_path,
                                                    'y_test.pkl'),
                                       'rb'))

        self.estimator = self.model.best_estimator_
        # self.pipeline = self.model.best_estimator_

        if model_name == 'Logistic Regression':
            self.model = self.estimator['logistic']
        elif model_name == 'Random Forest':
            self.model = self.estimator['random_forest']

        self.scaler = self.estimator['scaler']

        self.x_test_scaled = self.scaler.transform(self.x_test)

    def run(self) -> None:
        """
        This is the main function that runs all the functionality for the class
        :return: None
        """
        if self.model_name == 'Logistic Regression':
            if self.only_test:
                explainer_lr_test = shap.LinearExplainer(self.model, self.x_test_scaled)
                shap_values_lr_test = explainer_lr_test.shap_values(self.x_test_scaled)
                plt.figure(figsize=(5, 16))
                shap.summary_plot(shap_values_lr_test,
                                  self.x_test_scaled,
                                  feature_names=self.x_test.columns, show=False)
                st.pyplot(plt.gcf(), bbox_inches='tight', pad_inches=1)
                plt.clf()

        elif self.model_name == 'Random Forest':
            if self.only_test:
                rf_summary_plot(self.model, self.x_test_scaled, self.x_test.columns)
                # explainer_rf_test = shap.TreeExplainer(self.model)
                # shap_values_rf_test = explainer_rf_test.shap_values(self.x_test_scaled)
                # shap.summary_plot(shap_values_rf_test,
                #                   self.x_test_scaled,
                #                   feature_names=self.x_test.columns, show=False)
                #
                st.pyplot(plt.gcf(), bbox_inches='tight')
                plt.clf()

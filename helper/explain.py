import os
import pickle

import shap
import streamlit as st


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
        self.model_path = model_path
        self.data_path = data_path

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

        self.estimator = self.model.best_estimator_[-1]
        self.pipeline = self.model.best_estimator_[:-1]

    def run(self,
            nsample: int = 100) -> None:
        """
        This is the main function that runs all the functionality for the class
        :return: None
        """

        x_train_transformed = self.pipeline.transform(self.x_train)
        # y_train_transformed = self.pipeline.transform(self.y_train)

        x_test_transformed = self.pipeline.transform(self.x_test)
        # y_test_transformed = self.pipeline.transform(self.y_test)
        # tmp = self.estimator.coef_.shape

        # Todo: FIX THIS BUG!!! READ DOCUMENTATION!!
        explainer = shap.Explainer(self.model, x_train_transformed, link="logit")

        shap_values = explainer.shap_values(self.x_test)
        st.pyplot(shap.summary_plot(shap_values,
                                    self.x_train.columns,
                                    show=True))

        # show_plot(shap.force_plot(explainer.expected_value[0],
        #                           shap_values[0][0, :],
        #                           self.x_test.iloc[0, :],
        #                           link="logit",
        #                           show=True))

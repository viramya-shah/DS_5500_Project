import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.graph_objs import Pie, Layout, Figure
from plotly.offline import init_notebook_mode
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


class ChurnAnalysis:
    def __init__(self,
                 input_data_file_name: str = './data.csv',
                 data_input_path: str = './data/raw_data',
                 data_output_path: str = './data/output',
                 model_path: str = './data/models',
                 ) -> None:
        """
        Init the data path variables
        :param data_input_path:
        :param data_output_path:
        :param model_path:
        """
        self.data_input_path = data_input_path
        self.data_output_path = data_output_path
        self.model_path = model_path

        self.raw_data = pd.read_csv(os.path.join(self.data_input_path, input_data_file_name),
                                    low_memory=False
                                    )

    def show_col_name(self) -> str:
        return ", ".join(self.raw_data.columns.tolist())

    def eda(self) -> tuple:
        """
        This function is used to make plots between various data attributes
        :return: Plotly figures
        """

        # pie chart to show imbalance
        y = self.raw_data["Churn"].value_counts()
        _layout = Layout(title='Churn')
        _data = Pie(labels=self.raw_data['Churn'].unique(), values=y.values.tolist())
        figure_1 = Figure(data=[_data], layout=_layout)

        # contract vs monthly charges
        figure_2 = px.box(self.raw_data, x="Contract", y="MonthlyCharges")

        # tenure and monthly charges
        figure_3 = px.scatter(self.raw_data, x="tenure", y="MonthlyCharges",
                              color='Churn', facet_col="Contract", facet_col_wrap=3,
                              title="Churn Rate Analysis")

        # Todo: Add comment
        figure_4 = px.scatter(self.raw_data,
                              x="tenure", y="MonthlyCharges",
                              color="Churn", marginal_y="rug",
                              marginal_x="histogram")

        return figure_1, figure_2, figure_3, figure_4


class TrainUtil:
    def __init__(self, dataset):
        self.data = dataset

    def preprocessed(self):
        """
        Reads the data frame and returns a categorical data frame 
        :param: dataset
        :returns: Preprocessed dataframe
        """
        catColumn = self.data[
            ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
             'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
             'Contract', 'PaperlessBilling', 'PaymentMethod']]
        categorical_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                            'InternetService',
                            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                            'StreamingMovies',
                            'Contract', 'PaperlessBilling', 'PaymentMethod']

        le = LabelEncoder()

        catColumn[categorical_cols] = catColumn[categorical_cols].apply(lambda col: le.fit_transform(col))
        catColumn['TotalCharges'] = self.data['TotalCharges']
        catColumn['MonthlyCharges'] = self.data['MonthlyCharges']
        bins = [0, 20, 40, 60, 80, 100]
        labels = [1, 2, 3, 4, 5]
        catColumn['tenure'] = pd.cut(self.data['tenure'], bins, labels=labels)
        return catColumn

    def split_dataset(self, features):
        """
        Reads the data frame converts the target variable into categorical and splits the data to trainset and testset 
        :param: features
        :returns: trainset (X_train, y_train) and testset (X_test, y_test)
        """
        lb_target = LabelEncoder()
        self.data['Churn'] = lb_target.fit_transform(self.data['Churn'])
        features['tenure'] = lb_target.fit_transform(features['tenure'])
        features['TotalCharges'] = pd.to_numeric(features['TotalCharges'], errors='coerce')
        features = np.nan_to_num(features)
        X_train, X_test, y_train, y_test = train_test_split(features, self.data['Churn'], test_size=0.3, random_state=0)
        return X_train, X_test, y_train, y_test

    def LR_model(self, X_train, y_train):
        """
        Reads the trainset and returns Logistic Regression model
        :param: trainset(X_train, y_train)
        :returns: LR model
        """
        logistic = LogisticRegression(solver='lbfgs')
        pca = PCA()
        scaler = StandardScaler()
        steps_lr = [('scaler', scaler), ('pca', pca), ('logistic', logistic)]

        pipeline = Pipeline(steps_lr)
        n_components = [5, 10, 30]
        Cs = np.logspace(-4, 4, 3)
        estimator = GridSearchCV(pipeline,
                                 param_grid={
                                     'pca__n_components': n_components,
                                     'logistic__C': Cs}, cv=5, refit=True)
        lr_model = estimator.fit(X_train, y_train)
        joblib.dump(lr_model, 'LR_model.sav')
        return lr_model

    def RF_model(self, X_train, y_train):
        """
        Reads the trainset and returns Random Forest Classifier model
        :param: trainset(X_train, y_train)
        :returns: RF model
        """
        scaler = StandardScaler()
        rf = RandomForestClassifier()
        steps_rf = [('scaler', scaler), ('rf', rf)]

        pipeline_rf = Pipeline(steps_rf)
        max_depth = [50, 100]
        estimator_rf = GridSearchCV(pipeline_rf,
                                    param_grid={
                                        'rf__max_depth': max_depth}, cv=5, refit=True)
        rf_model = estimator_rf.fit(X_train, y_train)
        joblib.dump(rf_model, 'RF_model.sav')
        return rf_model

    def evaluation(self, model, X_test, y_test):
        """
        Reads the testset and returns evaluation score
        :param: testset
        :returns: Evaluation Plot
        """
        predictions = model.predict(X_test)
        print(classification_report(y_test, predictions))
        print("score = %3.2f" % model.score(X_test, y_test))

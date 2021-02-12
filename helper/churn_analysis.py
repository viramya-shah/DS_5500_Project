import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


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

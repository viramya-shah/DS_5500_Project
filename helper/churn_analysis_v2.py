import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE, ADASYN
from plotly.graph_objs import Pie, Layout, Figure
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
                 output_col: str = 'Churn'
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

        self.output_column = output_col
        self.feature_columns = list(set(self.raw_data.columns) - set(self.output_column))

    def show_col_name(self) -> str:
        return ", ".join(self.raw_data.columns.tolist())

    def eda(self) -> tuple:
        """
        This function is used to make plots between various data attributes
        :return: Plotly figures
        """

        # # pie chart to show imbalance
        # y = self.raw_data["Churn"].value_counts()
        # _layout = Layout(title='Churn')
        # _data = Pie(labels=self.raw_data['Churn'].unique(), values=y.values.tolist())
        # figure_1 = Figure(data=[_data], layout=_layout)
        #
        # # contract vs monthly charges
        # figure_2 = px.box(self.raw_data, x="Contract", y="MonthlyCharges")
        #
        # # tenure and monthly charges
        # figure_3 = px.scatter(self.raw_data, x="tenure", y="MonthlyCharges",
        #                       color='Churn', facet_col="Contract", facet_col_wrap=3,
        #                       title="Churn Rate Analysis")
        #
        # # Todo: Add comment
        # figure_4 = px.scatter(self.raw_data,
        #                       x="tenure", y="MonthlyCharges",
        #                       color="Churn", marginal_y="rug",
        #                       marginal_x="histogram")
        # Pie Plot
        y = self.df["Churn"].value_counts()
        colors = ["blue", "#d62728"]
        fig = make_subplots(
            rows=6, cols=2,
            column_widths=[3.2, 3.2],
            row_heights=[200.4, 197.4, 0.1, 200.4, 0.1, 200.4],
            specs=[[{"type": "pie", "rowspan": 2}, {"type": "scatter"}],
                   [None, {"type": "scatter"}],
                   [None, None],
                   [{"type": "histogram"}, {"type": "bar"}]
                , [None, None], [{"type": "box"}, {"type": "box"}]],

            subplot_titles=("Churn", "Charges", "", "Tenure", "PaymentMethod", "Paperless Billing", "Contract"))

        fig.add_trace(go.Pie(
            labels=self.df['Churn'].unique(),
            values=y.values.tolist(),
            legendgroup="group",
            marker=dict(colors=colors),
            textinfo='percent+label'),
            row=1, col=1)

        # Scatter Plot
        bins = [0, 10, 20, 30, 40, 50, 60, 70, 80]
        labels = [0, 10, 20, 30, 40, 50, 60, 70]
        self.df['tenure_bin'] = pd.cut(self.df['tenure'], bins, labels=labels)

        # Monthly Charges
        df_MonthlyCharges = (self.df.groupby(['Churn', 'tenure_bin'])
                             .MonthlyCharges.mean()
                             .reset_index(name='avg_MonthlyCharges')

                             )
        data_MonthlyCharges_Yes = df_MonthlyCharges[df_MonthlyCharges.Churn == "Yes"]
        data_MonthlyCharges_No = df_MonthlyCharges[df_MonthlyCharges.Churn == "No"]

        fig.add_trace(
            go.Scatter(x=data_MonthlyCharges_Yes["tenure_bin"], y=data_MonthlyCharges_Yes["avg_MonthlyCharges"],
                       marker=dict(color="#d62728"), showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=data_MonthlyCharges_No["tenure_bin"], y=data_MonthlyCharges_No["avg_MonthlyCharges"],
                                 marker=dict(color="blue"), showlegend=False), row=1, col=2)
        fig.update_yaxes(title_text="MonthlyCharges", row=1, col=2)

        # Total Charges
        self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
        df_TotalCharges = (self.df.groupby(['Churn', 'tenure_bin'])
                           .TotalCharges.mean()
                           .reset_index(name='avg_TotalCharges')
                           )

        data_TotalCharges_Churn_Yes = df_TotalCharges[df_TotalCharges.Churn == "Yes"]
        data_TotalCharges_Churn_No = df_TotalCharges[df_TotalCharges.Churn == "No"]

        fig.add_trace(
            go.Scatter(x=data_TotalCharges_Churn_Yes["tenure_bin"], y=data_TotalCharges_Churn_Yes["avg_TotalCharges"],
                       marker=dict(color="#d62728"), showlegend=False), row=2, col=2)
        fig.add_trace(
            go.Scatter(x=data_TotalCharges_Churn_No["tenure_bin"], y=data_TotalCharges_Churn_No["avg_TotalCharges"],
                       marker=dict(color="blue"), showlegend=False), row=2, col=2)
        fig.update_xaxes(title_text="Tenure", range=[0, 80], row=2, col=2)
        fig.update_yaxes(title_text="TotalCharges", row=2, col=2)

        ## Histogram_Plot
        data_Churn_Yes = self.df[data.Churn == "Yes"]
        data_Churn_No = self.df[data.Churn == "No"]
        fig.add_trace(go.Histogram(
            x=data_Churn_Yes['tenure'],
            xbins=dict(
                start=0,
                end=100,
                size=10
            ),
            legendgroup="group", marker=dict(color="#d62728"), showlegend=False), row=4, col=1)
        fig.add_trace(go.Histogram(
            x=data_Churn_No['tenure'],
            xbins=dict(
                start=0,
                end=100,
                size=10
            ),
            legendgroup="group", marker=dict(color="blue"), showlegend=False), row=4, col=1)
        fig.update_xaxes(title_text="Tenure(Bin)", range=[0, 80], row=4, col=1)
        fig.update_yaxes(title_text="Count", row=4, col=1)

        # Bar Plot
        data_Churn_No_payment = data_Churn_No.groupby('PaymentMethod').size().sort_values(ascending=False).reset_index(
            name='Count')
        data_Churn_Yes_payment = data_Churn_Yes.groupby('PaymentMethod').size().sort_values(
            ascending=False).reset_index(name='Count')
        fig.add_trace(go.Bar(
            x=data_Churn_No_payment['PaymentMethod'], y=data_Churn_No_payment['Count'],
            marker=dict(color="blue"), showlegend=False), row=4, col=2)
        fig.add_trace(go.Bar(
            x=data_Churn_Yes_payment['PaymentMethod'], y=data_Churn_Yes_payment['Count'],
            marker=dict(color="#d62728"), showlegend=False), row=4, col=2)

        # BoxPlot
        data_Churn_No_billing = data_Churn_No.groupby('PaperlessBilling').size().sort_values(
            ascending=False).reset_index(name='Count_Pa')
        data_Churn_Yes_billing = data_Churn_Yes.groupby('PaperlessBilling').size().sort_values(
            ascending=False).reset_index(name='Count_Pa')
        fig.add_trace(go.Bar(
            x=data_Churn_No_billing['PaperlessBilling'], y=data_Churn_No_billing['Count_Pa'],
            legendgroup="group", marker=dict(color="blue"), showlegend=False), row=6, col=1)
        fig.add_trace(go.Bar(
            x=data_Churn_Yes_billing['PaperlessBilling'], y=data_Churn_Yes_billing['Count_Pa'],
            legendgroup="group", marker=dict(color="#d62728"), showlegend=False), row=6, col=1)
        fig.update_yaxes(title_text="Count", row=6, col=1)

        data_Churn_No_contract = data_Churn_No.groupby('Contract').size().sort_values(ascending=False).reset_index(
            name='Count_Pa')
        data_Churn_Yes_contract = data_Churn_Yes.groupby('Contract').size().sort_values(ascending=False).reset_index(
            name='Count_Pa')

        fig.add_trace(go.Bar(
            x=data_Churn_No_contract['Contract'], y=data_Churn_No_contract['Count_Pa'],
            legendgroup="group", marker=dict(color="blue"), showlegend=False), row=6, col=2)
        fig.add_trace(go.Bar(
            x=data_Churn_Yes_contract['Contract'], y=data_Churn_Yes_contract['Count_Pa'],
            legendgroup="group", marker=dict(color="#d62728"), showlegend=False), row=6, col=2)
        fig.update_yaxes(title_text="Count", row=6, col=2)

        fig.update_layout(
            autosize=False,
            width=1000,
            height=1000,
            margin=dict(
                l=50,
                r=50,
                b=100,
                t=100,
                pad=4
            ),
            paper_bgcolor="white",
        )

        return fig

    def _preprocess(self,
                    bins=5,
                    labels=None
                    ) -> None:
        """
        Encodes the categorical data and bins the continuous variables
        :param: bins: bins to divide the tenure into
        :param: labels: labels to give to the binned tenure data
        :return:
        """
        if labels is None:
            labels = [1, 2, 3, 4, 5]

        catColumn = self.raw_data[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                                   'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                                   'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                                   'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']]

        categorical_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                            'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']

        le = LabelEncoder()
        catColumn[categorical_cols] = catColumn[categorical_cols].apply(lambda col: le.fit_transform(col))
        catColumn['TotalCharges'] = self.raw_data['TotalCharges']
        catColumn['MonthlyCharges'] = self.raw_data['MonthlyCharges']
        catColumn['tenure'] = pd.cut(self.raw_data['tenure'],
                                     bins=bins,
                                     labels=labels,
                                     right=True,
                                     include_lowest=True)

        self.raw_data = catColumn.copy()

        self.raw_data['tenure'] = self.raw_data['tenure'].fillna(1).astype(float)
        self.raw_data['TotalCharges'] = pd.to_numeric(self.raw_data['TotalCharges'], errors='coerce').fillna(0).astype(
            float)
        self.raw_data['MonthlyCharges'] = self.raw_data['MonthlyCharges'].fillna(1).astype(float)

    def _check_imbalance(self, method: str = 'SMOTE', random_seed: int = 1769) -> dict:
        """
        This function checks for imbalance. Further, it resamples the data and return the dataframe.
        Currently, we are only using Oversampling.
        :param: method: This defines the type of sampling to be done. Possible values: ['SMOTE', 'RANDOM']
        :return: None
        """
        output = self.raw_data[self.output_column]
        self.feature_columns = list(set(self.raw_data.columns) - set(self.output_column))
        features = self.raw_data[self.feature_columns]

        before_sampling = Counter(self.raw_data[self.output_column])

        if method == 'SMOTE':
            sampler = SMOTE(sampling_strategy='auto',
                            random_state=random_seed,
                            n_jobs=-1)
        elif method == 'ADASYN':
            sampler = ADASYN(sampling_strategy='auto',
                             random_state=random_seed,
                             n_jobs=-1)
        else:
            sampler = RandomOverSampler(sampling_strategy='auto',
                                        random_state=random_seed)

        features_resampled, output_resampled = sampler.fit_resample(features, output)
        after_sampling = Counter(output_resampled)

        return {
            'before_sampling_counter': before_sampling,
            'after_sampling_counter': after_sampling,
            'feature_data': features_resampled,
            'output_resampled': output_resampled
        }

    def _split_dataset(self,
                       features: pd.DataFrame,
                       output: pd.DataFrame,
                       test_size: float = 0.2,
                       random_state: int = 1769) -> dict:
        """
        Splits the data to train set and test set
        :param: test_size: test size percent
        :returns: Train set (X_train, y_train) and Test set (X_test, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(features,
                                                            output,
                                                            test_size=test_size,
                                                            random_state=random_state)
        if os.path.join(os.path.join(self.data_output_path)):
            os.makedirs(os.path.join(self.data_output_path))

        pickle.dump(X_train, open(os.path.join(self.data_output_path, 'x_train.pkl'), 'wb'))
        pickle.dump(X_test, open(os.path.join(self.data_output_path, 'x_test.pkl'), 'wb'))
        pickle.dump(y_train, open(os.path.join(self.data_output_path, 'y_train.pkl'), 'wb'))
        pickle.dump(y_test, open(os.path.join(self.data_output_path, 'y_test.pkl'), 'wb'))

        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }

    def _model(self,
               X_train, y_train,
               model_name: str = 'Logistic Regression',
               apply_reduction: bool = False) -> str:
        """
        Trains the models, saves the pickled object to the model_path.
        :param: X_train: Training feature data
        :param: y_train: Output feature data
        :param model_name: which model to be used?
        :param apply_reduction: whether to apply PCA reduction
        :return: Classification report!
        """
        param_grid = {}
        n_components = [5, 10, 30]  # for PCA
        max_depth = [20, 50, 100]  # for RandomForest
        C = np.logspace(-4, 4, 3)  # For LR

        steps = [('scaler', StandardScaler())]

        if apply_reduction:
            steps.append(('dimension_reduction', PCA()))
            param_grid['dimension_reduction__n_components'] = n_components

        if model_name == 'Logistic Regression':
            steps.append(('logistic', LogisticRegression(solver='lbfgs')))
            param_grid['logistic__C'] = C

        elif model_name == 'Random Forest':
            steps.append(('random_forest', RandomForestClassifier()))
            param_grid['random_forest__max_depth'] = max_depth

        estimator = GridSearchCV(Pipeline(steps),
                                 param_grid=param_grid,
                                 cv=5,
                                 refit=True)

        estimator.fit(X_train, y_train)

        if not os.path.exists(os.path.join(self.model_path, model_name)):
            os.makedirs(os.path.join(self.model_path, model_name))

        pickle.dump(estimator,
                    open(os.path.join(self.model_path,
                                      model_name,
                                      f"{model_name}.pkl"),
                         'wb')
                    )
        predictions = estimator.predict(X_train)
        return classification_report(y_train, predictions)

    def predict(self,
                model_name,
                X_test,
                y_test) -> str:
        """

        :param model_name: Model name to be load
        :param X_test: Test feature data
        :param y_test: Test output data
        :return: Classification report
        """
        estimator = pickle.load(open(os.path.join(self.model_path,
                                                  model_name,
                                                  f"{model_name}.pkl"),
                                     'rb'))

        predictions = estimator.predict(X_test)
        return classification_report(y_test, predictions)

    def run(self,
            model_name: str = 'Logistic Regression',
            apply_reduction: bool = True) -> dict:
        """

        :param model_name:
        :param apply_reduction:
        :return:
        """
        self._preprocess(bins=5,
                         labels=[1, 2, 3, 4, 5])

        imbalance_dict = self._check_imbalance()
        split_dict = self._split_dataset(imbalance_dict['feature_data'],
                                         imbalance_dict['output_resampled'])

        train_report = self._model(split_dict['X_train'], split_dict['y_train'], model_name, apply_reduction)
        test_report = self.predict(model_name, split_dict['X_test'], split_dict['y_test'])

        return {
            'train_report': train_report,
            'test_report': test_report
        }
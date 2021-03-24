import pandas as pd
import numpy as np
import os
import plotly
from plotly.graph_objs import Pie, Layout,Figure
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)
from plotly.offline import plot, iplot
from data_util import DataUtil

class EDAUtil:
    def __init__(self, data):
        self.df = data

    def plot(self):
        """
        Reads the data frame and plots the necessary visualization
        :returns: Plotly plots
        """
        y = self.df["Churn"].value_counts()
        _layout = Layout(title='Churn')
        _data = Pie(labels = self.df['Churn'].unique(), values=y.values.tolist())
        fig = Figure(data=[_data], layout=_layout)
        fig.show()
        
        fig = px.box(self.df, x="Contract", y="MonthlyCharges")
        fig.show()
        
        fig = px.scatter(self.df, x="tenure", y="MonthlyCharges", 
                 color='Churn',
                 facet_col="Contract", facet_col_wrap=3,
                 title= "Churn Rate Analysis")
        fig.show()
        
        fig = px.scatter(self.df, 
                 x="tenure", y="MonthlyCharges", 
                 color="Churn", 
                 marginal_y="rug", marginal_x="histogram")
        fig.show()


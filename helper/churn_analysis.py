import pandas as pd
import plotly
import matplotlib.pyplot as plt
from helper.data_util import DataUtil


class ChurnAnalysis:
    def __init__(self, data_folder: str = './data/raw_data'):
        """
        Init the important variables and store the data variables
        """
        self.data_folder = data_folder
        self.model = None
        pass

    def _preprocess(self, ):
        """

        :return:
        """
        dataUtil = DataUtil()
        self.data_frame = dataUtil.run()
        # sklearn pipeline
        # return pipeline object (also has model)
        pass

    def _eda(self):
        """

        :return:
        """
        pass

    def _train(self):
        """

        :return:
        """
        # pipeline = self._preprocess()
        # pipeline.fit()
        pass

    def _hyper_parameter_search(self):
        """

        :return:
        """
        pass

    def run(self, save: bool = True, output_path: str = "./data/output_data"):
        """

        :param save:
        :param output_path:
        :return:
        """
        self._eda()
        self._train()
        self._hyper_parameter_search()
        pass

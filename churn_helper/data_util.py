import os

import pandas as pd


class DataUtil:
    def __init__(self, data_folder="./raw_data"):
        self.data_folder = data_folder

    def read_csv(self, file_name: str = "./data.csv", low_memory: bool = False) -> pd.DataFrame:
        """
        Reads the raw data csv and returns a Pandas dataframe
        :param: file_name: The file to be read
        :param: low_memory: memory usage for the dataframe
        :returns: Pandas' dataframe
        """
        return pd.read_csv(os.path.join(self.data_folder, file_name), low_memory=low_memory)

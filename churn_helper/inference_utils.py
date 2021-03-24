import os
from churn_helper.extra_utils import model_path


class InferenceUtils:
    def __init__(self):
        """
        This class is provides functionality for combining and inferring results from Churn and NLP module
        """
        if not os.path.exists(**model_path):
            raise ValueError
        pass

    def run(self):
        pass

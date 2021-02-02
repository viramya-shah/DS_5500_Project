import os
from helper.extra_utils import model_path


class InferenceUtils:
    def __init__(self):
        if not os.path.exists(**model_path):
            raise ValueError
        pass

    def run(self):
        pass

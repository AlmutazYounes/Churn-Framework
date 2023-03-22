from Utils.config import Config
from automl.gridSearch import GridSearch


class AutoML:
    def __init__(self, train, test):
        obj = GridSearch("Output", threshold=0.52)
        obj.grid_search(train, test, Config.label_name)

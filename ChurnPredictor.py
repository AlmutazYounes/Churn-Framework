import pandas as pd
from automl import AutoML
from feature_loader import FeatureLoader
from util import Util
from config import Config


class ChurnPredictor:
    def __init__(self):
        self.feature_definitions = Util.get_feature_definitions(Config.features_type)

    def load_data(self):
        self.train_data, self.test_data = Util.load_data(Config.features_path)

    def extract_features(self):
        feature_extractor = FeatureLoader(self.feature_definitions)
        self.train_data, self.test_data = feature_extractor.extract_features(self.train_data, self.test_data)
        self.train_data, self.test_data = feature_extractor.preprocessing(self.train_data, self.test_data)
        feature_extractor.save_features(self.train_data, self.test_data)

    def train_model(self):
        automl = AutoML()
        # self.model = automl.base_model(self.train_data, self.test_data)
        self.model = automl.train_model(self.train_data, self.test_data)

    # def predict(self, new_data):
    #     feature_extractor = FeatureLoader(self.feature_definitions)
    #     new_features = feature_extractor.extract_features(new_data)
    #     return self.model.predict(new_features)


    def run(self):
        self.load_data()
        print("Train Length: ", len(self.train_data))
        print("Test Length: ", len(self.test_data))

        self.extract_features()
        self.train_model()


if __name__ == '__main__':
    churn_predictor = ChurnPredictor()
    churn_predictor.run()
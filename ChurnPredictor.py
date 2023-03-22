from automl import Automl
from automl.Automl import AutoML
from Utils.config import Config, run_params
from features.feature_loader import FeatureLoader
from Utils.util import Util

class ChurnPredictor:
    def __init__(self):
        self.feature_definitions = Util.get_feature_definitions(Config.features_type)

    def load_data(self):
        self.train_data, self.test_data = Util.load_data(Config.features_path)

    def extract_features(self, sampling_method, ratio):
        feature_extractor = FeatureLoader(self.feature_definitions, sampling_method, ratio)
        self.train_data, self.test_data = feature_extractor.preprocessing(self.train_data, self.test_data)

        self.train_data, self.test_data = feature_extractor.extract_features(self.train_data, self.test_data)
        feature_extractor.save_features(self.train_data, self.test_data)
        return self.train_data, self.test_data

    def train_model(self, train_data, test_data):

        AutoML(train_data, test_data)
        # automl = AutoML()
        # self.model = automl.train_model(self.train_data, self.test_data)
        # self.model = automl.base_model(self.train_data, self.test_data)

    # def predict(self, new_data):
    #     feature_extractor = FeatureLoader(self.feature_definitions)
    #     new_features = feature_extractor.extract_features(new_data)
    #     return self.model.predict(new_features)

    def run(self):
        self.load_data()
        # print("Train Length: ", len(self.train_data))
        # print("Test Length: ", len(self.test_data))

        train_data, test_data = self.extract_features(None, 0)
        self.train_model(train_data, test_data)
        # for sampling_method, ratio_list in run_params.sampling.items():
        #     for ratio in ratio_list:
        #         print(f" ##################################### {sampling_method} : {ratio} ##################################### ")
        #         self.load_data()
        #         # print("Train Length: ", len(self.train_data))
        #         # print("Test Length: ", len(self.test_data))
        #
        #         self.extract_features(sampling_method, ratio)
        #         self.train_model()


if __name__ == '__main__':
    churn_predictor = ChurnPredictor()
    churn_predictor.run()
# To Do:
# Work on sampling teq.
# Handle missing values
# Apply some analysis on the resutls
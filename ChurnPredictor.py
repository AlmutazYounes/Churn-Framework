from Utils.config import Config, run_params
from Utils.util import Util
from features.feature_loader import FeatureLoader
from automl.H2oAutoMl import AutoML
# from automl.Automl import AutoML


class ChurnPredictor:
    def __init__(self):
        self.feature_definitions = Util.get_feature_definitions(Config.features_type)

    def extract_features(self, sampling_method, ratio):
        self.train_data, self.test_data = Util.load_data(Config.features_path)
        feature_extractor = FeatureLoader(self.feature_definitions, sampling_method, ratio)
        self.train_data, self.test_data = feature_extractor.preprocessing(self.train_data, self.test_data)

        self.train_data, self.test_data = feature_extractor.extract_features(self.train_data, self.test_data)
        feature_extractor.save_features(self.train_data, self.test_data, sampling_method, ratio)
        return self.train_data, self.test_data

    def train_model(self, train_data, test_data, sampling_method, ratio):
        AutoML(train_data, test_data, f"Output/{sampling_method}_{ratio}").fit()

    def run(self):
        for sampling_method, ratio_list in run_params.sampling.items():
            for ratio in ratio_list:
                print(
                    f" ##################################### {sampling_method} : {ratio} ##################################### ")
                train_data, test_data = self.extract_features(sampling_method, ratio)
                self.train_model(train_data, test_data, sampling_method, ratio)


if __name__ == '__main__':
    churn_predictor = ChurnPredictor()
    churn_predictor.run()


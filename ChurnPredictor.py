import logging

from features.feature_loader import FeatureLoader
from automl.H2oAutoMl import AutoML
from Utils.config import Config, run_params
from Utils.util import Util

class ChurnPredictor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.feature_definitions = Util.get_feature_definitions(Config.features_type)
        self.train_data = None
        self.test_data = None

    def extract_features(self, sampling_method, ratio):
        self.train_data, self.test_data = Util.load_data(Config.dataset_path)
        feature_extractor = FeatureLoader(self.feature_definitions, sampling_method, ratio)
        self.train_data, self.test_data = feature_extractor.preprocess_data(self.train_data, self.test_data)
        self.train_data, self.test_data = feature_extractor.extract_features(self.train_data, self.test_data)
        feature_extractor.save_features(self.train_data, self.test_data)
        self.logger.info(f"Features extracted using {sampling_method} and ratio {ratio}")

    def train_model(self, sampling_method, ratio):
        model = AutoML(self.train_data, self.test_data, f"Output/{sampling_method}_{ratio}")
        model.fit()
        self.logger.info(f"Model trained using {sampling_method} and ratio {ratio}")

    def run(self):
        for sampling_method, ratio_list in run_params.sampling.items():
            for ratio in ratio_list:
                self.logger.info(
                    f"##################################### {sampling_method} : {ratio} #####################################")
                self.extract_features(sampling_method, ratio)
                self.train_model(sampling_method, ratio)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    churn_predictor = ChurnPredictor()
    churn_predictor.run()

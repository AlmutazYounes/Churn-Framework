import numpy as np
import pandas as pd

from Utils.config import Config, run_params
from Utils.util import Util
from features.feature_engineer import FeatureEngineer


class FeatureLoader:
    def __init__(self, feature_definitions, sampling_method, sample_ratio):
        self.feature_definitions = feature_definitions
        self.sampling_method = sampling_method
        self.sample_ratio = sample_ratio

    def feature_extraction_methods(self, data, feature, type_, **conf):
        if type_ == "absolute":
            return Util.absolute(data, feature, **conf)
        if type_ == "categorical_encoding":
            return Util.categorical_encoding(data, feature, **conf)
        if type_ == "label":
            return Util.label(data, feature, **conf)

    def extract_features(self, train_data, test_data):
        train_features = pd.DataFrame()
        test_features = pd.DataFrame()
        # Train Data
        for feature in self.feature_definitions:
            feature_params = self.feature_definitions[feature]
            conf = self.feature_definitions[feature]["conf"]
            train_feature = self.feature_extraction_methods(train_data, feature, feature_params["type"], **conf)
            train_features = pd.concat([train_features, pd.DataFrame(train_feature)], axis=1)

            # Test Data
            test_feature = self.feature_extraction_methods(test_data, feature, feature_params["type"], **conf)
            test_features = pd.concat([test_features, pd.DataFrame(test_feature)], axis=1)

        fe = FeatureEngineer(self.sampling_method, self.sample_ratio, run_params.missing_values_numarical)
        train_features = fe.preprocess(train_features, self.feature_definitions, data_type="train", sampleing=True)
        test_features = fe.preprocess(test_features, self.feature_definitions, data_type="test", sampleing=False)
        return train_features, test_features

    def preprocessing(self, train_data, test_data):
        train_data = train_data.replace(" ", np.nan)
        test_data = test_data.replace(" ", np.nan)
        return train_data, test_data

    def save_features(self, train_data, test_data, sampling_method, ratio):
        train_data.to_csv(f"{Config.features_train_output_file}_{sampling_method}_{ratio}.csv", index=False)
        test_data.to_csv(f"{Config.features_test_output_file}_{sampling_method}_{ratio}.csv", index=False)

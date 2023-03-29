import numpy as np
import pandas as pd

from Utils.config import Config, run_params
from Utils.util import Util
from features.feature_engineer import FeatureEngineer

class FeatureLoader:
    def __init__(self, feature_defs, sampling_method, sample_ratio):
        """
        Initializes a FeatureLoader object with feature definitions, sampling method, and sample ratio.
        """
        self.feature_defs = feature_defs
        self.sampling_method = sampling_method
        self.sample_ratio = sample_ratio

    def extract_feature(self, data, feature, feature_type, **conf):
        """
        Extracts a specific feature from the given data using the appropriate feature extraction method.
        """
        if feature_type == "absolute":
            return Util.absolute(data, feature, **conf)
        if feature_type == "categorical_encoding":
            return Util.categorical_encoding(data, feature, **conf)
        if feature_type == "label":
            return Util.label(data, feature, **conf)

    def extract_features(self, train_data, test_data):
        """
        Extracts all the defined features from the given train and test data.
        """
        train_features = pd.DataFrame()
        test_features = pd.DataFrame()
        for feature in self.feature_defs:
            feature_params = self.feature_defs[feature]
            conf = feature_params["conf"]
            train_feature = self.extract_feature(train_data, feature, feature_params["type"], **conf)
            train_features = pd.concat([train_features, pd.DataFrame(train_feature)], axis=1)

            test_feature = self.extract_feature(test_data, feature, feature_params["type"], **conf)
            test_features = pd.concat([test_features, pd.DataFrame(test_feature)], axis=1)

        fe = FeatureEngineer(self.sampling_method, self.sample_ratio, run_params.missing_values_numarical)
        train_features = fe.preprocess(train_features, self.feature_defs, data_type="train", sampleing=True)
        test_features = fe.preprocess(test_features, self.feature_defs, data_type="test", sampleing=False)
        return train_features, test_features

    def preprocess_data(self, train_data, test_data):
        """
        Preprocesses the given train and test data by replacing any space characters with NaN values.
        """
        train_data = train_data.replace(" ", np.nan)
        test_data = test_data.replace(" ", np.nan)
        return train_data, test_data

    def save_features(self, train_data, test_data):
        """
        Saves the extracted train and test features to CSV files using the defined naming convention.
        """
        train_file_name = f"{Config.features_train_output_file}_{self.sampling_method}_{self.sample_ratio}.csv"
        test_file_name = f"{Config.features_test_output_file}_{self.sampling_method}_{self.sample_ratio}.csv"
        train_data.to_csv(train_file_name, index=False)
        test_data.to_csv(test_file_name, index=False)

import pandas as pd
import numpy as np
from util import Util
from config import Config

class FeatureLoader:
    def __init__(self, feature_definitions):
        self.feature_definitions = feature_definitions

    def feature_extraction_methods(self, data, feature, function_, **params):
        if function_ == "absolute":
            return Util.absolute(data, feature)
        if function_ == "categorical_encoding":
            return Util.categorical_encoding(data, feature)
        if function_ == "label":
            return Util.label(data, feature)

    def extract_features(self, train_data, test_data):
        train_features = pd.DataFrame()
        test_features = pd.DataFrame()
        # Train Data
        for feature in self.feature_definitions:
            feature_params = self.feature_definitions[feature]
            train_feature = self.feature_extraction_methods(train_data, feature, feature_params["function"])
            train_features = pd.concat([train_features, pd.DataFrame(train_feature)], axis=1)

        # Test Data
            test_feature = self.feature_extraction_methods(test_data, feature, feature_params["function"])
            test_features = pd.concat([test_features, pd.DataFrame(test_feature)], axis=1)

        return train_features, test_features

    def preprocessing(self, train_data, test_data):
        train_data = train_data.replace(" ", np.nan)
        test_data = test_data.replace(" ", np.nan)
        return train_data, test_data

    def save_features(self, train_data, test_data):
        train_data.to_csv(Config.features_train_output_file, index=False)
        test_data.to_csv(Config.features_test_output_file, index=False)

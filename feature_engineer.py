import os
import pickle

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class FeatureEngineer:
    def __init__(self):
        self.encoders = {}

    def OneHotEncoderStrategy(self, data, feature, data_type):
        if data_type == "train":
            encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
            encoded_data = encoder.fit_transform(data[[feature]])
            with open(f"encoders/onehot_encoder_{feature}.pkl", "wb") as f:
                pickle.dump(encoder, f)
        else:
            with open(f"encoders/onehot_encoder_{feature}.pkl", "rb") as f:
                encoder = pickle.load(f)
            encoded_data = encoder.transform(data[[feature]])

        encoded_features = pd.DataFrame(encoded_data,
                                        columns=[f"{feature}_{cat}" for cat in encoder.categories_[0]])
        return encoded_features

    def encode_binary_feature(self, data, feature, binary_map):

        return data[feature].map(binary_map)

    def preprocess(self, data, feature_definitions, data_type="train"):
        encoded_features = []
        for feature in feature_definitions:
            if feature_definitions[feature]["preprocess"] == "one_hot_encoding":
                encoded_vals = self.OneHotEncoderStrategy(data, feature, data_type)
                encoded_features.append(encoded_vals)
                data = data.drop(columns=[feature])
            if feature_definitions[feature]["preprocess"] == "binary_encoding":
                encoded_vals = self.encode_binary_feature(data, feature, feature_definitions[feature]["binary_map"])
                encoded_features.append(encoded_vals)
                data = data.drop(columns=[feature])

        data_encoded = pd.concat(encoded_features, axis=1)

        return pd.concat([data, pd.DataFrame(data_encoded)], axis=1)

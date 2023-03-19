import json

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from config import Config


class Util:
    @staticmethod
    def absolute(data, feature_name):
        return data[[feature_name]].reset_index(drop=True)

    @staticmethod
    def categorical_encoding(data, feature_name):
        label_encoder = LabelEncoder()
        data[feature_name] = label_encoder.fit_transform(data[feature_name])
        one_hot_encoder = OneHotEncoder(sparse_output=False)
        encoded_data = one_hot_encoder.fit_transform(data[[feature_name]])
        num_cols = encoded_data.shape[1]
        col_names = [f"{feature_name}{i + 1}" for i in range(num_cols)]
        return pd.DataFrame(encoded_data, columns=col_names)

    @staticmethod
    def label(data, feature_name):
        label_encoder = LabelEncoder()
        encoded_data = label_encoder.fit_transform(data[feature_name])
        return pd.DataFrame(encoded_data, columns=[feature_name])

    @staticmethod
    def load_data(data_path, test_size=Config.test_size, random_state=Config.random_state):
        df = pd.read_csv(data_path)
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)
        return train_data, test_data

    @staticmethod
    def get_feature_definitions(features_type):
        if features_type == "telecom":
            with open("telecom_feature_definitions.json", 'r') as f:
                feature_definitions = json.load(f)
            return feature_definitions

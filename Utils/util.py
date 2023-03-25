import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from Utils.config import Config

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
            with open(Config.telecom_data, 'r') as f:
                feature_definitions = json.load(f)
            return feature_definitions

    @staticmethod
    def check_path(path):
        if path == "" : return
        if not os.path.exists(path):
            Util.check_path("/".join(path.split("/")[:-1]))
            if '.' not in path.split("/")[-1]:
                os.mkdir(path)

    @staticmethod
    def knn_impute(df, col, n_neighbors=3):
        """
        Impute missing categorical values using KNN.

        Parameters:
        df (pd.DataFrame): The dataset with missing categorical values.
        col (str): The column name to impute missing values.
        n_neighbors (int): The number of nearest neighbors to use for imputation (default 3).

        Returns:
        pd.DataFrame: The dataset with imputed missing values.
        """
        # Select columns with no missing values to use as features
        features = df.loc[:, df.columns != col].dropna()

        # Get the indexes of rows with missing values for the target column
        missing_idx = df.index[df[col].isnull()].tolist()

        # Compute distances between rows using Jaccard distance
        distances = pdist(features.apply(lambda x: x.astype('category').cat.codes), Util.jaccard)

        # Impute missing values using KNN
        imputed = df.copy()
        for idx in missing_idx:
            # Get the indices of the nearest neighbors
            nearest = np.argsort(distances[idx])[:n_neighbors]
            nearest_values = df.loc[nearest, col].values

            # Find the most common value among the neighbors
            mode = pd.Series(nearest_values).mode()
            if len(mode) > 0:
                imputed.at[idx, col] = mode[0]

        return imputed

    @staticmethod
    def jaccard(u, v):
        """
        Compute the Jaccard distance between two arrays of binary values.

        Parameters:
        u (np.array): The first array.
        v (np.array): The second array.

        Returns:
        float: The Jaccard distance between the arrays.
        """
        return 1 - np.sum(u & v) / np.sum(u | v)

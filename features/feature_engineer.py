import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, ClusterCentroids
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler

from Utils.config import Config


class FeatureEngineer:
    def __init__(self, sampling_method, ratio):
        self.scalers_encoders = dict()
        # self.samplers = dict()
        self.sampling_method = sampling_method
        self.sampler_list = {"RandomOverSampler": RandomOverSampler(sampling_strategy=ratio),
                             "RandomUnderSampler": RandomUnderSampler(sampling_strategy=ratio),
                             "SMOTE": SMOTE(sampling_strategy=0.5),
                             "TomekLinks": TomekLinks(),
                             "ClusterCentroids": ClusterCentroids(),
                             "SMOTETomek": SMOTETomek()
                             }

    def OneHotEncoderStrategy(self, data, feature, data_type):
        if data_type == "train":
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            encoded_data = encoder.fit_transform(data[[feature]])
            self.scalers_encoders[f"one_hot_encoding_{feature}_encoder"] = encoder
        else:
            try:
                encoder = self.scalers_encoders[f"one_hot_encoding_{feature}_encoder"]
                encoded_data = encoder.transform(data[[feature]])
            except KeyError:
                return None

        encoded_features = pd.DataFrame(encoded_data,
                                        columns=[f"{feature}_{cat}" for cat in encoder.categories_[0]])
        return encoded_features

    def ScalerStrategy(self, data, feature, data_type, scaler_type):
        if data_type == "train":
            scaler = scaler_type()
            scaled_data = scaler.fit_transform(data[[feature]])
            self.scalers_encoders[f"{scaler_type.__name__.lower()}_{feature}_scaler"] = scaler
        else:
            try:
                scaler = self.scalers_encoders[f"{scaler_type.__name__.lower()}_{feature}_scaler"]
                scaled_data = scaler.transform(data[[feature]])
            except KeyError:
                return None

        scaled_features = pd.DataFrame(scaled_data, columns=[f"{feature}_scaled"])
        return scaled_features

    def LabelEncoderStrategy(self, data, feature, data_type):
        if data_type == "train":
            encoder = LabelEncoder()
            encoded_data = encoder.fit_transform(data[feature])
            self.scalers_encoders[f"label_encoding_{feature}_encoder"] = encoder
        else:
            try:
                encoder = self.scalers_encoders[f"label_encoding_{feature}_encoder"]
                encoded_data = encoder.transform(data[feature])
            except KeyError:
                return None

        encoded_features = pd.DataFrame(encoded_data, columns=[f"{feature}_encoded"])
        return encoded_features

    def encode_binary_feature(self, data, feature, binary_map):
        return data[feature].map(binary_map)

    def preprocess(self, data, feature_definitions, data_type, sampleing=False):
        # Initialize empty lists to store encoded, scaled, and transformed features
        encoded_features = []
        scaled_features = []
        transformed_features = []
        features_to_remove = []

        # Loop over each feature in feature_definitions
        for feature in feature_definitions:
            preprocess_type = feature_definitions[feature]["preprocess"]

            # One-hot encoding
            if preprocess_type == "one_hot_encoding":
                features_to_remove.append(feature)
                encoded_vals = self.OneHotEncoderStrategy(data, feature, data_type)
                if encoded_vals is not None:
                    encoded_features.append(encoded_vals)

            # Binary encoding
            elif preprocess_type == "binary_encoding":
                features_to_remove.append(feature)
                encoded_vals = self.encode_binary_feature(data, feature, feature_definitions[feature]["binary_map"])
                if encoded_vals is not None:
                    encoded_features.append(encoded_vals)

            # Standard scaling
            elif preprocess_type == "standard_scaling":
                features_to_remove.append(feature)
                scaled_vals = self.ScalerStrategy(data, feature, data_type, StandardScaler)
                if scaled_vals is not None:
                    scaled_features.append(scaled_vals)

            # Min-max scaling
            elif preprocess_type == "minmax_scaling":
                features_to_remove.append(feature)
                scaled_vals = self.ScalerStrategy(data, feature, data_type, MinMaxScaler)
                if scaled_vals is not None:
                    scaled_features.append(scaled_vals)

            # RobustScaler scaling
            elif preprocess_type == "RobustScaler_scaling":
                features_to_remove.append(feature)
                scaled_vals = self.ScalerStrategy(data, feature, data_type, RobustScaler)
                if scaled_vals is not None:
                    scaled_features.append(scaled_vals)

            # Label encoding
            elif preprocess_type == "label_encoding":
                features_to_remove.append(feature)
                encoded_vals = self.LabelEncoderStrategy(data, feature, data_type)
                if encoded_vals is not None:
                    encoded_features.append(encoded_vals)

            # # If the preprocess type is not recognized, raise an error
            # else:
            #     assert(f"Unrecognized preprocess type: {preprocess_type} for feature {feature}")

        # Combine encoded, scaled, and transformed features into a single DataFrame
        if encoded_features:
            encoded_df = pd.concat(encoded_features, axis=1)
        else:
            encoded_df = pd.DataFrame()

        if scaled_features:
            scaled_df = pd.concat(scaled_features, axis=1)
        else:
            scaled_df = pd.DataFrame()

        if transformed_features:
            transformed_df = pd.concat(transformed_features, axis=1)
        else:
            transformed_df = pd.DataFrame()

        # Combine all features into a single DataFrame
        processed_data = pd.concat([data, encoded_df, scaled_df, transformed_df], axis=1)
        processed_data = processed_data.drop(columns=features_to_remove)

        # # Apply sampling method if specified
        # if self.sampling_method and sampleing:
        #     sampled_data, sampled_target = self.sampler_list[self.sampling_method].fit_resample(processed_data.drop(columns=[Config.label_name]), processed_data[Config.label_name])
        #     sampled_data = pd.DataFrame(sampled_data, columns=sampled_data.columns)
        #     target = pd.Series(sampled_target, name=Config.label_name)
        #     processed_data = pd.concat([sampled_data, target], axis=1)

        return processed_data

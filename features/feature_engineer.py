import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, ClusterCentroids
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler

from Utils.config import Config
from Utils.util import Util


class FeatureEngineer:
    def __init__(self, sampling_method, ratio, missing_values_numerical):
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
        self.missing_values_numerical = missing_values_numerical

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

    def handle_missing_values_numerical(self, processed_data):
        """
        Handle missing numerical values in a pandas DataFrame.

        Args:
            processed_data (pandas.DataFrame): DataFrame containing numerical data with missing values.

        Returns:
            pandas.DataFrame: DataFrame with missing values handled according to the specified method.

        """
        # Define the handling method based on the value of self.missing_values_numerical
        if self.missing_values_numerical == "drop":
            # Drop any rows with missing values
            processed_data = processed_data.dropna()
            return processed_data
        elif self.missing_values_numerical == "regression":
            # Create a copy of the DataFrame to impute missing values
            df_imputed = processed_data.copy()
            # Get the names of columns with missing values
            missing_cols = processed_data.columns[processed_data.isnull().any()]
            for col in missing_cols:
                # Drop any rows with missing values in the current column
                df_temp = df_imputed.dropna(subset=[col])
                # Split the DataFrame into feature and target arrays
                X = df_temp.drop(columns=[col])
                y = df_temp[col]
                # Fit a linear regression model to predict the missing values
                model = LinearRegression()
                model.fit(X, y)
                # Identify the rows with missing values in the current column
                missing_values = df_imputed[col].isnull()
                # Use the trained model to predict the missing values and replace them in the DataFrame
                df_imputed.loc[missing_values, col] = model.predict(df_imputed[missing_values].drop(columns=[col]))
            return df_imputed
        elif self.missing_values_numerical == "auto":
            # Create a copy of the DataFrame to impute missing values
            df_imputed = processed_data.copy()
            # Get the names of columns with missing values
            missing_cols = processed_data.columns[processed_data.isnull().any()]
            for col in missing_cols:
                # Calculate the percentage of missing values in the current column
                missing_pct = df_imputed[col].isnull().sum() / len(df_imputed)
                if missing_pct == 1:
                    # If all values are missing, drop the column
                    df_imputed = df_imputed.drop(columns=[col])
                elif missing_pct > 0 and missing_pct <= 0.1:
                    # If there are fewer than 10% missing values, drop the rows with missing values
                    df_imputed = df_imputed.dropna(subset=[col])
                else:
                    # If there are more than 10% missing values, check for correlations with other columns
                    corr_matrix = df_imputed.corr()
                    corr_with_missing_col = corr_matrix[col]
                    corr_with_missing_col.drop(col, inplace=True)
                    if any(abs(corr_with_missing_col) < 0.5):
                        # If the correlations are weak, impute the missing values with the mean
                        df_imputed[col].fillna(df_imputed[col].mean(), inplace=True)
                    else:
                        # If the correlations are strong, fit a linear regression model to predict the missing values
                        df_temp = df_imputed.dropna(subset=[col])
                        X = df_temp.drop(columns=[col])
                        y = df_temp[col]
                        model = LinearRegression()
                        model.fit(X, y)
                        missing_values = df_imputed[col].isnull()
                        df_imputed.loc[missing_values, col] = model.predict(
                            df_imputed[missing_values].drop(columns=[col]))
            return df_imputed

    def handle_missing_values_categorical(self, processed_data, drop_threshold=0.3):
        """
        Handle missing categorical values in the dataset.

        Parameters:
        processed_data (pd.DataFrame): The processed dataset with missing categorical values.
        drop_threshold (float): The threshold to drop columns with too many missing values (default 0.3).

        Returns:
        pd.DataFrame: The processed dataset with imputed missing values.
        """

        # Drop columns with too many missing values
        null_pct = processed_data.isnull().sum() / len(processed_data)
        drop_cols = null_pct[null_pct > drop_threshold].index.tolist()
        processed_data = processed_data.drop(columns=drop_cols)

        # Impute missing values using mode or KNN
        categorical_cols = processed_data.select_dtypes(include=['object', 'category']).columns.tolist()
        missing_cols = processed_data.columns[processed_data.isnull().any()].tolist()

        for col in missing_cols:
            if col in categorical_cols:
                if processed_data[col].isnull().sum() < len(processed_data) * 0.1:
                    processed_data[col].fillna(processed_data[col].mode()[0], inplace=True)
                else:
                    # Use KNN imputation for missing values in categorical columns
                    knn_imputed = Util.knn_impute(processed_data, col)
                    processed_data[col] = knn_imputed[col]

        return processed_data

    def apply_sampling(self, processed_data, sampleing):
        # Apply sampling method if specified
        if self.sampling_method and sampleing:
            sampled_data, sampled_target = self.sampler_list[self.sampling_method].fit_resample(
                processed_data.drop(columns=[Config.label_name]), processed_data[Config.label_name])
            sampled_data = pd.DataFrame(sampled_data, columns=sampled_data.columns)
            target = pd.Series(sampled_target, name=Config.label_name)
            processed_data = pd.concat([sampled_data, target], axis=1)
        return processed_data

    def preprocess(self, data, feature_definitions, data_type, sampleing=False):

        # Handle missing categorical values
        data = self.handle_missing_values_categorical(data)
        # Missing Values numerical values
        data = self.handle_missing_values_numerical(data)

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

        # Sampling methods
        processed_data = self.apply_sampling(processed_data, sampleing)

        return processed_data

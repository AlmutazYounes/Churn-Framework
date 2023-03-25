class Config:
    features_path = r"dataset/WA_Fn-UseC_-Telco-Customer-Churn -missing.csv"
    features_type = "telecom"
    label_name = "Churn"
    features_train_output_file = "features/features_train.csv"
    features_test_output_file = "features/features_test.csv"
    telecom_data = "features/telecom_feature_definitions.json"
    test_size = 0.2
    random_state = 101

class run_params:
    missing_values_numarical = "auto" # "regression", "drop", "auto"
    sampling = {
        None: [0],
        # "RandomOverSampler": [0.6, 0.7, 0.8, 0.9, 1],
        # "RandomUnderSampler": [0.6, 0.7, 0.8, 0.9, 1],
        # "SMOTE": ["minority"],
        # "TomekLinks": [0],
        # "ClusterCentroids": [0],
        # "SMOTETomek": [0]
    }


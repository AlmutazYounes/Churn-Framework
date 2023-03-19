class Config:
    features_path = r"WA_Fn-UseC_-Telco-Customer-Churn.csv"
    features_type = "telecom"
    label_name = "Churn"
    features_train_output_file = "features/features_train.csv"
    features_test_output_file = "features/features_test.csv"
    test_size = 0.2
    random_state = 101

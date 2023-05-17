# Churn Prediction Framework

The **Churn Prediction Framework** is a machine learning tool designed to accept any kind of data with minimal changes required to the framework. It includes an AutoML capability and performs all preprocessing steps required to deal with churn prediction.

## Getting Started

To get started with the Churn Prediction Framework, you will need to follow the steps below:

### Installation

1. Clone this repository to your local machine.
2. Install the required dependencies using the following command:

pip install -r requirements.txt


### Usage

1. Prepare your dataset: Your dataset should include the target variable (i.e., whether a customer has churned or not) and any features you wish to use to predict churn.

2. Configure the parameters: Open the `config.yml` file and configure the parameters according to your dataset.

3. Run the `main.py` script: Run the following command to start the AutoML process and train a model:

python main.py

## Configuration

The `config.py` file contains the configuration parameters for the Churn Prediction Framework. You can customize these parameters according to your dataset. The following parameters can be configured:

- `dataset_path`: The path to the dataset file.
- `target_variable`: The name of the target variable in the dataset.
- `features_type`: The type of the data, for example "telecom".
- `features_train_output_file`: The path to the train file.
- `features_test_output_file`: The path to the test file.
- `telecom_data`: The json file used to extract the features from the dataset, this file needs customization based on the data and project types.
- `test_size`: test size.

## Contributing

Contributions to the Churn Prediction Framework are welcome. If you would like to contribute, please follow the steps below:

1. Fork this repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Push your changes to your fork.
5. Create a pull request.

## License

This project is licensed under the **MIT License** - see the `LICENSE` file for details.

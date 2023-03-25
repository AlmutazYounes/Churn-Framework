import h2o
import pandas as pd
from h2o.automl import H2OAutoML
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from Utils.config import Config


class AutoML:
    def __init__(self, train, test, output):
        self.train = train
        self.test = test
        self.output = output
        self.results = pd.DataFrame()

    def fit(self):
        h2o.init()

        # Convert data to H2OFrame
        train_h2o = h2o.H2OFrame(self.train)
        test_h2o = h2o.H2OFrame(self.test)

        # Identify predictors and response
        x = [i for i in train_h2o.columns if i != Config.label_name]
        y = Config.label_name

        # Convert the target column to categorical
        train_h2o[y] = train_h2o[y].asfactor()
        test_h2o[y] = test_h2o[y].asfactor()

        # Define AutoML settings and include specific algorithms
        aml = H2OAutoML(max_models=50, seed=1, nfolds=5,
                        max_runtime_secs=3600, stopping_metric='AUTO',
                        sort_metric='f1',
                        include_algos=['GBM', 'DRF', 'GLM', 'DeepLearning'],
                        balance_classes=True,
                        class_sampling_factors=None  # Set this to a list of floats if needed
                        )

        # Train AutoML model
        aml.train(x=x, y=y, training_frame=train_h2o)

        # Get the leaderboard and select the best model
        leaderboard = aml.leaderboard
        best_model = h2o.get_model(leaderboard[0, 'model_id'])

        # Predict on the test set and calculate metrics
        y_pred = best_model.predict(test_h2o).as_data_frame()['predict']
        metrics = {
            'accuracy': accuracy_score,
            'precision': precision_score,
            'recall': recall_score,
            'f1': f1_score,
            'roc_auc': roc_auc_score,
            # Add more metrics here as desired
        }
        model_metrics = {'model': best_model.model_id}
        for metric_name, metric_func in metrics.items():
            model_metrics[metric_name] = metric_func(self.test[Config.label_name], y_pred)

        # Append the metrics of the best model to the results dataframe
        self.results = self.results.append(model_metrics, ignore_index=True)

        # Loop through all models in the leaderboard
        for index, row in leaderboard.as_data_frame().iterrows():
            # Get the model ID
            model_id = row['model_id']
            # Load the model
            model = h2o.get_model(model_id)
            # Predict on the test set and calculate metrics
            y_pred = model.predict(test_h2o).as_data_frame()['predict']
            model_metrics = {'model': model_id}
            for metric_name, metric_func in metrics.items():
                model_metrics[metric_name] = metric_func(self.test[Config.label_name], y_pred)
            # Append the model metrics to the results dataframe
            self.results = self.results.append(model_metrics, ignore_index=True)

        # Save the results to a CSV file
        self.results.to_csv(f"{self.output}.csv", index=False)

        h2o.cluster().shutdown()

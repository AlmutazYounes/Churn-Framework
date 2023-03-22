import itertools
import random
import pandas as pd

from Utils.util import Util
from automl.models import ModelName, Classifier, Parameters
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class GridSearch:
    def __init__(self, out_dir = "output", threshold = 0.5):
        self.list_models = [ModelName.LGBM,
                            ModelName.LogisticLR,
                            ModelName.DecisionTree,
                            ModelName.RandomForest]

        Util.check_path(out_dir)
        self.out_dir = out_dir

        header = ["model", "params", "precision", "recall", "accuracy", "f1"]
        self.final_df = pd.DataFrame(columns=header)
        self.final_df.to_csv(f"{out_dir}/results.csv", index=False)

        # filter out the models whose f1 score is below this threshold
        self.threshold = threshold
        self.best_acc = 0

    def predict(self, clf, x_test, y_test):
        y_pred = clf.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=1.0)
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        return dict([("accuracy",accuracy),("precision",precision), ("recall",recall), ("f1", f1)])

    def save(self, model_name, params, accuracies):
        if accuracies["f1"] < self.threshold: return

        if accuracies["f1"] > self.best_acc:
            self.best_acc = accuracies["f1"]
            result_str = f"Model: {model_name.name} || Accuracy: {accuracies['accuracy']}  || F1 score: {accuracies['f1']}"
            print(result_str)

        final_df = pd.read_csv(f"{self.out_dir}/results.csv")
        last_idx = final_df.tail(1).index.tolist()
        last_idx = last_idx[0]+1 if len(last_idx)>0 else 0
        final_df.loc[last_idx] = {**{"model":model_name.name, "params":params}, **accuracies}

        final_df = final_df.sort_values(by=["f1"], ascending=[False]).head(10)
        try:
            final_df.to_csv(f"{self.out_dir}/results.csv", index=False)
        except:
            print("results.csv file is in use")
            pass

    # train all possible combinations of models parameters
    def train_all_models(self, model_name, train_data, test_data, label):
        dict_parameters = Parameters.get_params(model_name)

        # build a list of all possible combinations of model parameters
        param_values = list(dict_parameters.values())
        values_tuples = itertools.product(*param_values)
        param_names = list(dict_parameters.keys())
        list_params = [dict(zip(param_names, tuple)) for tuple in values_tuples]
        random.shuffle(list_params)

        x_train = train_data.drop(columns=[label])
        x_test = test_data.drop(columns=[label])
        y_train, y_test = train_data[label], test_data[label]

        for params in list_params:
            clf = Classifier.train(model_name, x_train, y_train, params)
            if clf == "error":
                continue
            accuracies = self.predict(clf, x_test, y_test)
            self.save(model_name, params, accuracies)

    # build a list of all possible combinations of model's parameters
    def grid_search(self, train_data, test_data, label):
        for model_name in self.list_models:
            self.train_all_models(model_name, train_data, test_data, label)



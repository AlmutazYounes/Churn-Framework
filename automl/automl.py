import h2o
from h2o.automl import H2OAutoML

import pandas as pd
from Utils.config import Config
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from tqdm import tqdm

class AutoML:
    def __init__(self, train, test, output):
        self.train_data = train
        self.test_data = test
        self.output = output
        self.results = pd.DataFrame()
        self.models = [
            ("DecisionTreeClassifier", DecisionTreeClassifier(), {
                'max_features': ['sqrt'],
                'criterion': ['gini', 'entropy'],
                'splitter': ['best', 'random'],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }),
            ("RandomForestClassifier", RandomForestClassifier(), {
                'max_depth': [3],
                'max_features': ['sqrt'],
                'criterion': ['gini', 'entropy'],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'n_estimators': [50, 100, 200]
            }),
            # ("SVM", SVC(), {
            #     'C': [0.1],
            #     'gamma': ['scale'],
            #     'kernel': ['linear']
            # }),
            ("LogisticRegression", LogisticRegression(), {
                'C': [0.1, 1, 10],
                'penalty': ['l2'],
                'max_iter' : [10000],
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
            }),
            ("LGBMClassifier", LGBMClassifier(), {
                'num_leaves': [31],
                'max_depth': [-1, 5, 10],
                'learning_rate': [0.01, 0.1, 1],
                'n_estimators': [50, 100, 200],
                'boosting_type': ['gbdt', 'dart'],

            })
        ]

    def fit(self):
        # drop any rows with missing values from the train data
        self.train_data = self.train_data.dropna()
        self.test_data = self.test_data.dropna()

        # separate the train_features and train_labels back into separate DataFrames
        train_features = self.train_data.drop(columns=Config.label_name)
        train_labels = self.train_data[Config.label_name].ravel()

        test_features = self.test_data.drop(columns=Config.label_name)
        test_labels = self.test_data[Config.label_name].ravel()

        results = []
        best_acc = 0.0  # keep track of the best accuracy so far
        for name, clf, param_grid in tqdm(self.models):

            grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='f1')
            grid_search.fit(train_features, train_labels)

            best_clf = grid_search.best_estimator_
            best_params = grid_search.best_params_

            # make predictions on the testing data
            test_preds = best_clf.predict(test_features)

            # calculate evaluation metrics
            acc = accuracy_score(test_labels, test_preds)
            f1 = f1_score(test_labels, test_preds, average='weighted')
            recall = recall_score(test_labels, test_preds, average='weighted')
            precision = precision_score(test_labels, test_preds, average='weighted')

            results.append({
                "Model": name,
                "Best Parameters": best_params,
                "Accuracy": acc,
                "F1 Score": f1,
                "Recall": recall,
                "Precision": precision
            })

            if acc > best_acc:
                # update best_acc and print the table
                best_acc = acc
                df_results = pd.DataFrame(results)
                # format Best Parameters column
                df_results["Best Parameters"] = df_results["Best Parameters"].apply(
                    lambda x: str(x).replace('{', '').replace('}', ''))
                df_results["Best Parameters"] = df_results["Best Parameters"].apply(
                    lambda x: x.replace("'", '').replace(", ", '\n'))
                # sort results by Accuracy in descending order
                df_results = df_results.sort_values(by='F1 Score', ascending=False).reset_index(drop=True)
                # print the table
                print()
                print(df_results.to_markdown(index=False))

        # create pandas DataFrame for all results
        df_results = pd.DataFrame(results)
        # format Best Parameters column
        # df_results["Best Parameters"] = df_results["Best Parameters"].apply(
        #     lambda x: str(x).replace('{', '').replace('}', ''))
        # df_results["Best Parameters"] = df_results["Best Parameters"].apply(
        #     lambda x: x.replace("'", '').replace(", ", '\n'))
        # sort results by Accuracy in descending order
        df_results = df_results.sort_values(by='F1 Score', ascending=False).reset_index(drop=True)
        df_results.to_csv(f"{self.output}.csv", index=False)

        # print the final table
        print('\nBest model:')
        print(df_results.head(1).to_markdown(index=False))
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from config import Config
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


class AutoML:
    def __init__(self):
        self.models = [
            ("DecisionTreeClassifier", DecisionTreeClassifier(), {
                'max_depth': [3, 5, None],
                'max_features': ['sqrt', 'log2', None]
            }),
            ("RandomForestClassifier", RandomForestClassifier(), {
                'n_estimators': [100, 500],
                'max_depth': [3, 5, None],
                'max_features': ['sqrt', 'log2', None]
            }),
            ("SVM", SVC(), {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto'],
                'kernel': ['linear', 'rbf']
            }),
            ("LogisticRegression", LogisticRegression(), {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
            }),
            ("LGBMClassifier", LGBMClassifier(), {
                'num_leaves': [31, 64, 128],
                'max_depth': [-1, 3, 5],
                'learning_rate': [0.1, 0.01, 0.001],
                'n_estimators': [100, 500]
            })
        ]

    def train_model(self, train_data, test_data):
        # drop any rows with missing values from the train data
        train_data = train_data.dropna()
        test_data = test_data.dropna()

        # separate the train_features and train_labels back into separate DataFrames
        train_features = train_data.drop(columns=Config.label_name)
        train_labels = train_data[[Config.label_name]]

        test_features = test_data.drop(columns=Config.label_name)
        test_labels = test_data[[Config.label_name]]

        results = []
        best_acc = 0.0  # keep track of the best accuracy so far
        for name, clf, param_grid in tqdm(self.models):

            grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
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
                df_results = df_results.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
                # print the table
                print()
                print(df_results.to_markdown(index=False))

        # create pandas DataFrame for all results
        df_results = pd.DataFrame(results)
        # format Best Parameters column
        df_results["Best Parameters"] = df_results["Best Parameters"].apply(
            lambda x: str(x).replace('{', '').replace('}', ''))
        df_results["Best Parameters"] = df_results["Best Parameters"].apply(
            lambda x: x.replace("'", '').replace(", ", '\n'))
        # sort results by Accuracy in descending order
        df_results = df_results.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)

        # print the final table
        print('\nBest model:')
        print(df_results.head(1).to_markdown(index=False))

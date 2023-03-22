from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from enum import Enum


class ModelName(Enum):
    RandomForest = 1
    LogisticLR = 2
    DecisionTree = 3
    LGBM = 4


class Parameters:
    @staticmethod
    def get_params( model_name):
        obj = Parameters()
        if model_name == ModelName.LGBM: return obj.lgbm()
        if model_name == ModelName.RandomForest: return obj.rf()
        if model_name == ModelName.LogisticLR: return obj.llr()
        if model_name == ModelName.DecisionTree: return obj.dt()

    def lgbm(self):
        return {
            'objective': ['binary'],#, 'multiclass', 'regression'],
            'boosting_type': ['gbdt', 'dart', 'goss'],
            'num_leaves': [31, 63, 127, 255],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [-1, 5, 10, 20],
            'n_estimators': [50, 100, 200],
            'min_child_samples': [20, 50, 100],
            'subsample': [0.5, 0.8, 1],
            'colsample_bytree': [0.5, 0.8, 1],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [0, 0.1, 0.5],
            'importance_type': ['split', 'gain']
        }

    def rf(self):
        return {
            'n_estimators': [50, 100, 200],
            'max_features': ['auto', 'sqrt', 'log2', None],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }


    def dt(self):
        return {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        }

    def llr(self):
        return {
            'penalty': ['l1', 'l2'],
            'C': [0.1, 1, 10],
            'solver': ['liblinear', 'saga']
        }


class Classifier:
    @staticmethod
    def train(model_name, x, y, params):
        obj = Classifier()
        try:
            if model_name == ModelName.LGBM: return obj.lgbm(x, y, params)
            if model_name == ModelName.RandomForest: return obj.rf(x, y, params)
            if model_name == ModelName.LogisticLR: return obj.llr(x, y, params)
            if model_name == ModelName.DecisionTree: return obj.dt(x, y, params)
        except:
            return "error"

    def lgbm(self, x, y, params):
        clf = lgb.LGBMClassifier(**params)
        clf.fit(x, y)
        return clf

    def dt(self, x, y, params):
        clf = DecisionTreeClassifier(**params)
        clf.fit(x, y)
        return clf

    def llr(self, x, y, params):
        clf = LogisticRegression(**params)
        clf.fit(x, y)
        return clf

    def rf(self, x, y, params):
        clf = RandomForestClassifier(**params)
        clf.fit(x, y)
        return clf


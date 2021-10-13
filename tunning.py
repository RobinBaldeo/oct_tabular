import pandas as pd
import numpy as np
from optuna.integration import LightGBMPruningCallback, XGBoostPruningCallback

from catboost import CatBoostClassifier
import optuna
import sklearn.model_selection as ms
from sklearn.linear_model import LogisticRegression
pd.set_option('expand_frame_repr', False)
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb

class model_best_para:
    def __init__(self, xtrain, xval, ytrain, yval, num_iter):
        self.xtrain = xtrain
        self.xval = xval
        self.yval = yval
        self.ytrain = ytrain
        self.num_iter = num_iter

    def cat(self, trial, x, y):

        xtrain, xval = x
        ytrain, yval = y


        fixed_para_xgb = {
            "eval_set": [(xval, yval)],
            "early_stopping_rounds": 5,
            'verbose': 25,
        }

        param = {
            "eval_metric":"AUC",
            "objective": "Logloss",
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 1e-8, 0.1),
            "depth": trial.suggest_int("depth", 1, 5),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.3, .5),
            # "num_leaves" : trial.suggest_int('num_leaves', 1, 1000),
            "boosting_type": "Ordered",
            # "min_child_samples": trial.suggest_int('min_child_samples', 1, 300),
            "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bernoulli"]),
            "used_ram_limit": "3gb",
        }

        if param["bootstrap_type"] == "Bayesian":
            param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
        elif param["bootstrap_type"] == "Bernoulli":
            param["subsample"] = trial.suggest_float("subsample", 1e-8, 1)

        cat_ = CatBoostClassifier(**param)
        cat_.fit(xtrain, ytrain, **fixed_para_xgb)
        preds = cat_.predict_proba(np.float32(xval))[:, 1]

        return roc_auc_score(yval, preds)















    def xgb(self, trial, x, y):
        xtrain, xval = x
        ytrain, yval = y

        xval = np.float32(xval)
        xtrain = np.float32(xtrain)


        fixed_para_xgb = {
            "eval_set": [(xval, yval)],
            # 'eval_set': [ (dvalid, 'valid')],
            "early_stopping_rounds": 5,
            'eval_metric': 'auc',
            'verbose': 25,
            'callbacks': [XGBoostPruningCallback(trial, 'validation_0-auc')],
        }

        param_xgb = {
            "n_jobs":-1,
            "objective": "binary:logistic",
            # "eta":trial.suggest_float("eta", .01, .5, log=True),
            "max_depth":trial.suggest_int("max_depth", 2, 8),
            "n_estimators": trial.suggest_int("n_estimators", 200, 600),
            "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
            "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
            "subsample": trial.suggest_loguniform("subsample", 0.2, 1.0),
            "colsample_bytree": trial.suggest_loguniform("colsample_bytree", 0.2, 1.0),
            'use_label_encoder': False
        }

        xboo = xgb.XGBClassifier(**param_xgb)
        xboo.fit(xtrain, ytrain, **fixed_para_xgb)
        preds = xboo.predict_proba(np.float32(xval))[:, 1]

        return roc_auc_score(yval, preds)






    def light(self, trial, x, y):
        xtrain, xval = x
        ytrain, yval = y

        fixed_para = {
            "early_stopping_rounds": 5,
            'eval_metric': 'auc',
            'eval_set': [(xval, yval)],
            'verbose': 25,
            'callbacks': [LightGBMPruningCallback(trial, 'auc')],
        }

        param = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": 'rf',
            "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
            "max_depth":trial.suggest_int("max_depth", 1, 8),
            # "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 1.),
            "lambda_l1": trial.suggest_loguniform('lambda_l1', 1e-3, 10.0),
            "lambda_l2": trial.suggest_loguniform('lambda_l2', 1e-3, 10.0),
            "num_leaves": trial.suggest_int('num_leaves', 1, 1000),
            "min_child_samples": trial.suggest_int('min_child_samples', 1, 300),

            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7)
        }



        gbm = LGBMClassifier(**param)
        gbm.fit(xtrain, ytrain, **fixed_para)
        preds = gbm.predict_proba(xval)[:, 1]

        return roc_auc_score(yval, preds)

    def para(self):
        study = optuna.create_study(direction='maximize', study_name="robin")
        study.optimize(lambda i: self.cat(i, (self.xtrain, self.xval), (self.ytrain, self.yval)),
                       n_trials=self.num_iter)
        # study.optimize(lambda i: self.light(i, (self.xtrain, self.xval), (self.ytrain, self.yval)), n_trials=self.num_iter)
        # study.optimize(lambda i: self.xgb(i, (self.xtrain, self.xval), (self.ytrain, self.yval)), n_trials=self.num_iter)
        # print(f"best trial : {([j for i, j in study.best_trial.intermediate_values.items()])[-1]}")
        print()
        print(study.best_trial.params)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from  tunning import model_best_para
import sklearn.model_selection as ms
from sklearn.utils.fixes import loguniform
import warnings
warnings.simplefilter("ignore", UserWarning)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from lightgbm import LGBMClassifier
# from sklearn.model_selection import  RandomizedSearchCV, KFold

def reduce_df(df):

    print(f"orginal dataset :{df.memory_usage().sum() / 1024 ** 2} mb")
    for i in df.columns:
        col_type = df[i].dtypes

        if str(col_type)[0:1] in ["i", "f"]:
            col_min, col_max = np.min(df[i]), np.max(df[i])
            if str(col_type)[0:1] == "i":
                for j in [np.int8,np.int16,np.int32, np.int64]:
                    if col_min > np.iinfo(j).min and col_max < np.iinfo(j).max:
                        df[i] = df[i].astype(j)
                        break
            else:
                for j in [np.float16,np.float32,np.float64]:
                    if col_min > np.finfo(j).min and col_max < np.finfo(j).max:
                        df[i] = df[i].astype(j)
                        break

    print(f"dataset reduced to :{df.memory_usage().sum() / 1024 ** 2} mb")
    return df


def main():
    # train = reduce_df(pd.read_csv("train.csv").set_index("id"))
    # print(train.head(5))
    # train.to_pickle("train")
    #
    # test = reduce_df(pd.read_csv("test.csv").set_index("id"))
    # test.to_pickle("test")

    train = pd.read_pickle("train")
    # print(train.dtypes)
    # train.to_pickle("train")
    #
    # test = pd.read_pickle("test")
    # print(test.head(20))
    # train = train.sample(1000)

    y = train["target"]
    x = train.drop(columns = ["target"])

    # scaler = StandardScaler()
    # x = np.float32(scaler.fit_transform(x))

    xtrain, xval, ytrain, yval = ms.train_test_split(x, y, test_size=.2, shuffle=True, random_state=45)
    # lb = model_best_para(xtrain, xval, ytrain, yval, 10)
    # lb.para()

    # param = {'n_estimators': np.arange(50, 250, 50), 'learning_rate': loguniform(1e-4, 1e0)}
    # ad = ms.RandomizedSearchCV(AdaBoostClassifier(), param_distributions = param, n_jobs=-1)

    # ad = AdaBoostClassifier(**{'learning_rate': 0.032763180024690086, 'n_estimators': 200})
    ad = LGBMClassifier(**{'n_jobs':-1, 'n_estimators': 847, 'max_depth': 7, 'learning_rate': 0.04617423130344099, 'lambda_l1': 1.6029632425074436, 'lambda_l2': 0.0010928490115681689, 'num_leaves': 124, 'min_child_samples': 93, 'feature_fraction': 0.7917548593828119, 'bagging_fraction': 0.96375720421119, 'bagging_freq': 3})
    ad.fit(xtrain, ytrain)
    #
    # # print()
    # # print(ad.best_params_)
    # # print()
    # # print(ad.best_score_)
    #
    preds = ad.predict_proba(xval)[:, 1]
    score = roc_auc_score(yval, preds)
    print(score)













# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


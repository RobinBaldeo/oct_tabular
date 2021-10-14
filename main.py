
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
import warnings
warnings.simplefilter("ignore", UserWarning)
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from lightgbm import LGBMClassifier
# from sklearn.model_selection import  RandomizedSearchCV, KFold
from catboost import CatBoostClassifier
from sklearn.linear_model import SGDClassifier
from collections import namedtuple
import sys
np.set_printoptions(threshold=sys.maxsize)

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


def prePorocess():
    train = reduce_df(pd.read_csv("train.csv").set_index("id"))
    train.to_pickle("train")
    #
    test = reduce_df(pd.read_csv("test.csv"))
    test.to_pickle("test")

    print(test.head(5))
    print(train.head(5))


def main():


    train = pd.read_pickle("train")
    test_o = pd.read_pickle("test")



    # train = train.sample(1000)
    # test_o = test_o.sample(1000)



    y = train["target"]
    x = train.drop(columns = ["target"])
    test = test_o.drop(columns=["id"])


    folds = 10


    gbdt_para = {'n_jobs':-1, 'n_estimators': 847, 'max_depth': 7, 'learning_rate': 0.04617423130344099, 'lambda_l1': 1.6029632425074436, 'lambda_l2': 0.0010928490115681689, 'num_leaves': 124, 'min_child_samples': 93, 'feature_fraction': 0.7917548593828119, 'bagging_fraction': 0.96375720421119, 'bagging_freq': 3}
    goss_para = {'n_jobs':-1,'n_estimators': 888, 'max_depth': 3, 'lambda_l1': 0.045547053858182196, 'lambda_l2': 1.290891976923166, 'num_leaves': 614, 'min_child_samples': 261, 'min_child_weight': 15.811750102552908}
    cat_para = {'colsample_bylevel': 0.05606508594613661, 'depth': 4, 'learning_rate': 0.3840012528742531, 'bootstrap_type': 'Bernoulli', 'subsample': 0.645075461245303}

    gbdt_ = LGBMClassifier(**gbdt_para)
    goss_ = LGBMClassifier(**goss_para)
    cat_ = CatBoostClassifier(**cat_para)


    models_lst = []
    models = namedtuple("models", "ind type fit_")
    models_lst.append(models(ind = 0, type= "gbdt", fit_ = gbdt_))
    models_lst.append(models(ind = 1, type="goss", fit_=goss_))
    models_lst.append(models(ind = 2, type="cat", fit_=cat_))


    df_split = ms.StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    train_meta_x = np.zeros((len(train.index), 3))
    train_meta_y = np.zeros((len(train.index), 3))
    weights = np.zeros((folds, 3))
    fold_score = np.zeros((folds, 3))


    fold_pred_cv = np.zeros((len(test.index) , folds))
    fold_pred = np.zeros((len(test.index), 3))

    for m in models_lst:

        start = 0
        end = 0

        for counter, (trn, val) in enumerate(df_split.split(x, y)):

            end += len(val)
            mod_ = m.fit_.fit(x.iloc[trn, :].values, y.iloc[trn])
            meta_pred = mod_.predict_proba(x.iloc[val, :])[:, 1]
            fold_pred_cv[:, counter] = mod_.predict_proba(test.values)[:, 1]
            train_meta_x[start:end, m.ind] = meta_pred
            train_meta_y[start:end, m.ind] = y.iloc[val]

            weights[counter, m.ind] = roc_auc_score(y.iloc[val], meta_pred)
            fold_score[counter, m.ind] = weights[counter, m.ind]
            print(counter)

            if counter == folds -1:

                weights[:, m.ind] = weights[:, m.ind]/np.sum(weights[:, m.ind], axis=0)
                fold_pred[:,m.ind] = np.dot(fold_pred_cv,weights[:, m.ind])
            start += len(val)


    print(fold_score)

    train_meta_x = pd.DataFrame(train_meta_x)
    train_meta_x.to_pickle("train_meta_x")

    train_meta_y = pd.DataFrame(train_meta_y)
    train_meta_y.to_pickle("train_meta_y")

    fold_pred = pd.DataFrame(fold_pred )
    fold_pred.to_pickle("fold_pred")


    a = pd.read_pickle("train_meta_x")
    b = pd.read_pickle("train_meta_y")
    c = pd.read_pickle("fold_pred")
    print(a.head(5))
    print(b.head(5))
    print(c.head(5))


    # second_model2 = SGDClassifier(max_iter=10000, loss='log')
    # #
    # second_model2.fit(train_meta_x, train_meta_y[:, 0])
    # pred = second_model2.predict_proba(fold_pred)[:, 1]
    # # print(pred)
    # #
    # final = pd.read_csv("test.csv")
    # final = final.merge(pd.DataFrame(pred), right_index=True, left_index=True)
    # final.columns = ["id", "target"]
    # final.to_csv("sub_v1.csv", index=False)
        # print(base_meta)
        # print(weights)



















# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
    # prePorocess()


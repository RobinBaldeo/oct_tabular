
import pandas as pd
import numpy as np

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
    # train = reduce_df(pd.read_csv("train.csv"))
    # train.to_pickle("train")
    #
    # test = reduce_df(pd.read_csv("test.csv"))
    # test.to_pickle("test")

    train = pd.read_pickle("train")
    print(train.head(20))
    # train.to_pickle("train")
    #
    test = pd.read_pickle("test")
    print(test.head(20))

    # for
    # print(train.head(20))
    # print()
    # print(len(train.index))
    # print()
    # print(len(train[train.target == 0]))










# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


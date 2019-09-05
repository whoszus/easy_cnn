import torch.nn as nn
import pandas as pd
import torch
from sklearn import preprocessing
import time

embedding = nn.Embedding(728, 16)
col_names = ["city", "dev_name", "dev_type", "time", "alm_level"]

train_f = "data/data_train_2_sort_5w.csv"
test_f = "data/data_test_2_sort_5w.csv"


# log_f = open("logs/" + str(batch_x) + '_' + c_time + '.log', 'w+')


def data_encode(train_data_X):
    print("开始转换数据格式》...")
    for name in col_names:
        if name != 'time':
            le = preprocessing.LabelEncoder()
            le.fit(train_data_X[name])
            train_data_X[name] = le.transform(train_data_X[name])
    print(train_data_X.head(10), train_data_X.shape)
    return train_data_X


# 加载数据 &drop_duplicates
def load_csv_data(file):
    print("开始加载数据..")
    data = pd.read_csv(file, names=col_names, encoding='utf-8')
    data = data.drop_duplicates()
    # 去除连续的重复
    data = data.loc[(data['dev_name'].shift() != data['dev_name'])].dropna().reset_index(drop=True)
    data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S', infer_datetime_format=True, errors="raise")
    print("数据加载完毕，去重完毕，去重后数据量：%d" % len(data))
    return data

# def delete_duplicates(data):
#     for index, row in data.iterrows():
#         if index > 0 and data['dev_name'][index] == data['dev_name'][index]:
#             data.drop(index)
#     return data

# 加载数据
def load_data(data_type='train'):
    print("开始加载数据.....")
    data_train = load_csv_data(train_f)
    data_train = data_encode(data_train)

    data_val = load_csv_data(test_f)
    data_val = data_encode(data_val)

    data = {
        'train_data': data_train,
        'val_data': data_val
    }
    torch.save(data, 'pickle/data_6_5w.pt')
    print(data)
    return


if __name__ == '__main__':
    dev_name_embedding = load_data()
    # for i in range(10) :
    #     print(dev_name_embedding[0].clone())
    #     similarity, words = torch.topk(torch.mv(embedding.weight, dev_name_embedding[i].clone()), 5)
    #
    #     print( words,similarity)

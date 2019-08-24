import torch.nn as nn
import pandas as pd
import torch
from sklearn import preprocessing
import pickle

embedding = nn.Embedding(728, 16)
col_names = ["city", "dev_name", "dev_type", "time", "alm_level"]

train_f = "data/data_train_2_sort_5w.csv"
test_f = "./data/data_test_2_sort_5w.csv"


# log_f = open("logs/" + str(batch_x) + '_' + c_time + '.log', 'w+')


def data_encode(train_data_X):
    print("开始转换数据格式》...")
    x_les = []
    for name in col_names:
        le = preprocessing.LabelEncoder()
        le.fit(train_data_X[name])
        x_les.append(le)
        train_data_X[name] = le.transform(train_data_X[name])
        with open('pickle/name_pickle', 'wb') as f:
            pickle.dump(train_data_X[name], f, -1)
    print(train_data_X.head(10), train_data_X.shape)
    return train_data_X


# 加载数据 &drop_duplicates
def load_csv_data(file):
    print("开始加载数据..")
    data = pd.read_csv(file, names=col_names, encoding='utf-8')
    data = data.drop_duplicates().dropna().reset_index(drop=True)
    data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S', infer_datetime_format=True, errors="raise")
    print("数据加载完毕，去重完毕，去重后数据量：%d" % len(data))
    return data


# 加载数据
def load_data(data_type='train'):
    if data_type == 'train':
        print("开始加载数据.....")
        data = load_csv_data(train_f)
    else:
        print("进行测试.....")
        data = load_csv_data(test_f)
    # 按时间切分
    data = data_encode(data)
    dev_name = data['dev_name']
    dev_name_embedding = embedding(torch.tensor(dev_name.values,dtype=torch.long))
    return dev_name_embedding


if __name__ == '__main__':
    dev_name_embedding = load_data()
    # for i in range(10) :
    #     print(dev_name_embedding[0].clone())
    #     similarity, words = torch.topk(torch.mv(embedding.weight, dev_name_embedding[i].clone()), 5)
    #
    #     print( words,similarity)

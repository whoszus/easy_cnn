import torch.nn as nn
import pandas as pd
import torch
from sklearn import preprocessing
import os

col_names = ["city", "dev_name", "dev_type", "time", "alm_level"]

torch_save_path = 'data/csv/data_train_2_sort.torch'
file_path = 'data/csv/data_train_2_sort.csv'
save_pt_path = 'data/pt/'
train_rate = 0.8


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
def load_csv_data():
    if os.path.exists(torch_save_path):
        print("数据已结构化..读取数据中..")
        return torch.load(torch_save_path)['data_all']
    print("开始加载数据..")
    data = pd.read_csv(file_path, names=col_names, encoding='utf-8')
    data = data.drop_duplicates().dropna().reset_index(drop=True)
    # 去除连续的重复
    # data = data.loc[(data['dev_name'].shift() != data['dev_name'])].dropna().reset_index(drop=True)
    data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S', infer_datetime_format=True, errors="raise")
    torch_save_data = {
        'data_all': data
    }
    torch.save(torch_save_data, torch_save_path)
    print("数据加载完毕，去重完毕，去重后数据量：%d" % len(data))
    return data


def load_data_by_num(data, start, num):
    return data[data['time'] >= start][num]


def load_data_by_date(data, start, end):
    # return data[start <= data['time'] <= end]
    return data[[any([a, b]) for a, b in zip(data.time >= start, data.time <= end)]]


# 加载数据
def load_data(start, end, num=0):
    print("开始加载数据.....")
    data_load = load_csv_data()
    start_time = pd.to_datetime(start, format='%Y-%m-%d %H:%M:%S')
    end_time = pd.to_datetime(end, format='%Y-%m-%d %H:%M:%S')
    if num > 0:
        data_all = load_data_by_num(data_load, start_time, num)
    else:
        data_all = load_data_by_date(data_load, start_time, end_time)

    data_len = len(data_all['dev_name'])
    data_train = data_all[:int(train_rate * data_len)]
    data_train = data_encode(data_train)

    data_val = data_all[int((1 - train_rate) * data_len) * -1:]
    data_val = data_encode(data_val)

    data_voc = len(pd.unique(pd.array(data_train['dev_name']))) + len(pd.unique(pd.array(data_val['dev_name'])))

    data = {
        'train_data': data_train,
        'val_data': data_val,
        'voc': data_voc
    }
    torch.save(data, save_pt_path + start + '~' + end + '-' + str(data_voc) + '.pt')
    return


if __name__ == '__main__':
    start_time_str = '2018-06-01'
    end_time_str = '2018-06-15'

    load_data(start_time_str, end_time_str)

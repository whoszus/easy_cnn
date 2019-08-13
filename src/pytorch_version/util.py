import torch.nn as nn
import pandas as pd
import torch
from sklearn import preprocessing


embedding = nn.Embedding(728, 8)
col_names = ["dev_name", "time", "dev_type", "city", "alm_level"]
train_f = "data/data_1.csv"
test_f = "./data/test_1_8k.csv"
# log_f = open("logs/" + str(batch_x) + '_' + c_time + '.log', 'w+')



def data_encode(train_data_X):
    print("开始转换数据格式》...")
    x_les = []
    for name in col_names:
        le = preprocessing.LabelEncoder()
        le.fit(train_data_X[name])
        x_les.append(le)
        train_data_X[name] = le.transform(train_data_X[name])
    print("转换数据完毕》。", train_data_X.head(), train_data_X.shape)
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
    dev_name_embedding = embedding(torch.from_numpy(dev_name.values))
    return dev_name_embedding



if __name__ == '__main__':
    dev_name_embedding = load_data()

    similarity, words = torch.topk(torch.mv(embedding.weight, dev_name_embedding[0].clone()), 5)

    print(similarity, words)

# coding=utf-8
# -*w- coding utf-8 -*-

import pandas as pd
from sklearn import preprocessing
import pickle
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader, TensorDataset
import time
from NNModule import NetAY

LR = 0.01
EPOCH = 200

col_names = ["dev_name", "time", "dev_type", "city", "alm_level"]
need_data_changed = False
# LR = 0.001
BATCH_SIZE = 64
load_pickle_data = False
c_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
log_f = open("logs/" + c_time + '.log', 'w+')

train_f = 'data/data_1.csv'
test_f = "data/data_2.csv"
embedding = nn.Embedding(728, 16)
# embedding_time = nn.Embedding(512, 8)
batch_x = 128
batch_y = 64
verison = '1001'


class MyDataSet(Dataset):
    """ my dataset."""

    # Initialize your data, download, etc.
    def __init__(self):
        # 读取csv文件中的数据
        train_data_x, train_data_y_name, train_data_y_time, encode_y_name = load_data_final()
        self.train_data_x = train_data_x
        self.train_data_y_name = train_data_y_name
        self.train_data_y_time = train_data_y_time
        self.encode_y_name = encode_y_name
        self.len = len(self.train_data_x)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.train_data_x[index], \
               self.train_data_y_name[index], \
               self.train_data_y_time[index], \
               self.encode_y_name[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len


class TestDataSet(Dataset):
    """ my dataset."""

    # Initialize your data, download, etc.
    def __init__(self):
        # 读取csv文件中的数据
        train_data_x, train_data_y_name, train_data_y_time, encode_y_name = load_data_test()
        self.train_data_x = train_data_x
        self.train_data_y_name = train_data_y_name
        self.train_data_y_time = train_data_y_time
        self.encode_y_name = encode_y_name
        self.len = len(self.train_data_x)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.train_data_x[index], \
               self.train_data_y_name[index], \
               self.train_data_y_time[index], \
               self.encode_y_name[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len


# 加载数据
def load_data(batch_x, data_type='train'):
    if data_type == 'train':
        print("开始加载数据.....", file=log_f)
        data = load_csv_data(train_f)
    else:
        print("进行测试.....", file=log_f)
        data = load_csv_data(test_f)
    # 按时间切分
    data = time_split(data, batch_x)
    # 转化为标签数据
    encode_x = data_encode(data)

    # embedding
    dev_name = encode_x['dev_name']
    dev_name_embedding = embedd(torch.from_numpy(dev_name.values))
    time_array = np.array(encode_x['time'].values, np.float32, copy=False)
    # todo
    time_tensor = torch.from_numpy(time_array).view(time_array.size, 1)
    train_x = torch.cat((dev_name_embedding, time_tensor), 1)
    # group data
    group_data, group_data_name, group_data_time = data_reshape_step(train_x, batch_x=batch_x,
                                                                     batch_y=batch_y)
    return group_data, group_data_name, group_data_time, reshape_encode_data(dev_name)


# 加载数据 &drop_duplicates
def load_csv_data(file):
    print("开始加载数据..", file=log_f)
    data = pd.read_csv(file, names=col_names, encoding='utf-8')
    data = data.drop_duplicates().dropna().reset_index(drop=True)
    data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S', infer_datetime_format=True,
                                  errors="raise")
    print("数据加载完毕，去重完毕，去重后数据量：%d" % len(data), file=log_f)
    return data


# 将时间处理成时间间隔
def time_split(train_data_x, batch_x):
    print("开始处理时间格式...", file=log_f)
    print(type(train_data_x))
    c_time = train_data_x['time']
    r_time = []
    for index, value in c_time.iteritems():
        try:
            if index == 0:
                r_time.append(0)
            else:
                get_sec = lambda x, y: (x - y).seconds if x > y else (y - x).seconds
                get_sec_min = lambda x: x if x < 1000 else 1000

                seconds = get_sec(c_time[index], c_time[index - 1])
                if seconds > 900:
                    print(c_time[index], c_time[index - 1], (c_time[index] - c_time[index - 1]).seconds, seconds)
                r_time.append(seconds)
        except:
            print(index, c_time[index], c_time[index - 1])
        # len_x += 1
        # except:
        #     print(c_time[index + 1],c_time[index])
        # if seconds > 1000:
        #     print(c_time[index], c_time[index + 1], seconds)
        #     print(index)
        # r_time.append(seconds)
    train_data_x['time'] = r_time
    print("处理时间格式完毕..", train_data_x.head(), file=log_f)
    return train_data_x


def embedd(input_data_x, type='dev_name'):
    if type == 'dev_name':
        output_x = embedding(input_data_x)
        # todo save embedding entiy
        # else:
        # output_x = embedding_time(input_data_x)
        return output_x


# LabelEncoder
#  >>> le = preprocessing.LabelEncoder()
#     >>> le.fit(["paris", "paris", "tokyo", "amsterdam"])
#     LabelEncoder()
#     >>> list(le.classes_)
#     ['amsterdam', 'paris', 'tokyo']
#     >>> le.transform(["tokyo", "tokyo", "paris"]) #doctest: +ELLIPSIS
#     array([2, 2, 1]...)
#     >>> list(le.inverse_transform([2, 2, 1]))
#     ['tokyo', 'tokyo', 'paris']、
# 转化为标签数据
def data_encode(train_data_X):
    print("开始转换数据格式》...", file=log_f)
    x_les = []
    for name in col_names:
        if name != 'time':
            le = preprocessing.LabelEncoder()
            le.fit(train_data_X[name])
            x_les.append(le)
            train_data_X[name] = le.transform(train_data_X[name])
    # dict
    with open('pickle/les.pickle', 'wb') as feature:
        pickle.dump(x_les, feature, -1)
    print("转换数据完毕》。", train_data_X.head(), train_data_X.shape, file=log_f)
    return train_data_X


# 重制数据格式为 【batch,5,64】
def data_reshape(train_data_x, batch_x, batch_y):
    print("开始组装数据..", file=log_f)
    tmp = []
    group_data = []
    tmp_y = []
    group_data_name = []
    group_data_time = []
    train_data_x = np.array(train_data_x)
    [rows, cols] = train_data_x.shape
    for i in range(rows):
        tmp.append(train_data_x[i])
        tmp_y.append(train_data_x[i])
        if (i + 1) % batch_x == 0:
            group_data.append(torch.tensor(tmp))
            tmp = []
        if (i + 1) % (batch_x + batch_y) == 0:
            data_y = tmp_y[batch_y * -1:]
            data_y = np.array(data_y)
            data_y_name = data_y[:, 0]
            data_y_time = data_y[:, 1]
            group_data_name.append(torch.tensor(data_y_name, dtype=torch.long))
            group_data_time.append(torch.tensor(data_y_time, dtype=torch.long))
    print("数据组装完毕...", file=log_f)
    return group_data, group_data_name, group_data_time


def reshape_encode_data(encode_data, step_i=12):
    encode_data = np.array(encode_data)
    rows = encode_data.shape[0]
    i = 1
    current_i = 1
    tmp = []
    group_data = []
    while i < rows:
        tmp.append(encode_data[i])
        i += 1
        if len(tmp) % (batch_x + batch_y) == 0:
            encode_y = tmp[batch_y * -1:]
            group_data.append(torch.tensor(np.array(encode_y)))
            current_i += step_i
            i = current_i
            tmp = []

    return group_data


# 重制数据格式为
def data_reshape_step(train_data_x, batch_x, batch_y, step_i=12):
    if not step_i:
        return data_reshape(train_data_x, batch_x, batch_y)

    print("开始组装数据..步长", step_i, file=log_f)
    tmp = []
    group_data = []
    tmp_y = []
    group_data_name = []
    group_data_time = []
    train_data_x = train_data_x.detach().numpy()
    [rows, cols] = train_data_x.shape
    current_i = 1
    i = 0
    while i < rows:
        tmp.append(train_data_x[i])
        tmp_y.append(train_data_x[i])
        i += 1
        if len(tmp) % batch_x == 0:
            group_data.append(torch.tensor(tmp))
            tmp = []
        if len(tmp_y) % (batch_x + batch_y) == 0:
            data_y = tmp_y[batch_y * -1:]  # 将倒数batch_y 条放入data_y
            data_y = np.array(data_y)

            data_y_name = data_y[:, 0:16]  # 第一列放入data_y_name

            data_y_time = data_y[:, 16] / 1000  # 第二列放入data_y_time 归一化
            group_data_name.append(
                torch.tensor(data_y_name, dtype=torch.float32))  # 每batch_y条作为一个group 放入group_data_name
            group_data_time.append(torch.tensor(data_y_time))
            current_i += step_i
            i = current_i
            print("当前装载进度：", current_i, file=log_f)
            tmp = []
            tmp_y = []

    group_data.pop(-1)  # 最后多出来一条没有对应的Y 值
    print("数据组装完毕...", time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()), file=log_f)
    print("数据增加后数量....", len(group_data_name), file=log_f)
    return group_data, group_data_name, group_data_time


# 暂未使用
def calculate_outdim(input_dim):
    return 50 if input_dim > 50 else (input_dim + 1) / 2


def pickle_loader(input):
    item = pickle.load(open(input, 'rb'))
    return item


# todo
def get_accuracy(module, epoch):
    # print(module) 开始进行测试
    test_data_set = TestDataSet()

    test_loader = torch.utils.data.DataLoader(dataset=test_data_set, batch_size=1, shuffle=True)
    best = 0.00
    for step, data in enumerate(test_loader):
        test_data_x, test_data_y_name, test_data_y_time, encode_y_name = data

        test_output = module(test_data_x.view(1, 1, 128, 17))
        name_res = test_output[0].view(64, 16)
        time_res = test_output[1].view(64)

        result_n = []
        result_n_s = []
        for name in name_res:
            # 这一步操作是将embedding的数据类似翻译回来
            similarity, words = torch.topk(torch.mv(embedding.weight, name.clone()), 1)
            result_n.append(np.array(words))
            result_n_s.append(similarity.detach().numpy())

        name_acy = get_name_acy(result_n, result_n_s, encode_y_name)
        time_acy = get_tim_acy(time_res, test_data_y_time)
        if name_acy > best:
            best = name_acy
        if step > 50 and best< 0.1:
            break

        print(step, 'current epoch :%d ' % epoch, '| test accuracy_name: %.2f' % name_acy,
              'accuracy_time:%.2f' % time_acy)
        print(step, 'current epoch :%d ' % epoch, '| test accuracy_name: %.2f' % name_acy,
              'accuracy_time:%.2f' % time_acy, file=log_f)

    torch.save(module, "modules/tmp" + str(best) + ".pickle")


def get_name_acy(m_res, m_res_s, y):
    res = np.array(m_res)
    res_sim = np.array(m_res_s)  # 相似度
    y = np.array(y)
    print("预测结果：", res, res_sim, file=log_f)
    print("实际结果：", y, file=log_f)
    print("预测结果：", res)
    print("实际结果：", y)
    mg = np.intersect1d(res, y)
    print("交集：", mg)

    if len(mg) > 0:
        # 此部分代码是同个位置，不合理，应为出现就算
        # bc1 = np.bincount(res.flatten())
        # bc2 = np.bincount(y.flatten())
        # # 统计相同元素匹配个数
        # same_count_list = [min(bc1[x], bc2[x]) for x in mg]
        # same_count = sum(same_count_list)
        # print("交集：", mg, file=log_f)

        return float(len(mg) / np.unique(y).size())
    print("此轮无结果.......", file=log_f)
    return 0.00


# todo ; How to measure time accuracy
def get_tim_acy(m_res, y):
    return 1


def load_data_final():
    print("start learning")
    if load_pickle_data:
        train_data_X, train_data_y_name, train_data_y_time, encode_y_name, y_time = open('pickle/train_data.pickle',
                                                                                         'rb')
    else:
        train_data_X, train_data_y_name, train_data_y_time, encode_y_name = load_data(batch_x=batch_x)
        # pickle dump data
        # with open('pickle/train_data.pickle', 'wb')as f:
        #     pickle.dump((train_data_X, train_data_y_name, train_data_y_time), f, -1)
    return train_data_X, train_data_y_name, train_data_y_time, encode_y_name


def load_data_test():
    print("装载测试数据")
    return load_data(data_type='test', batch_x=batch_x)


if __name__ == '__main__':
    my_data_set = MyDataSet()
    train_loader = DataLoader(dataset=my_data_set, batch_size=64, shuffle=True, num_workers=16)

    # 开始训练
    cnn = NetAY()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.MSELoss()  # the target label is not one-hotted
    loss_func_name = nn.CrossEntropyLoss()
    for epoch in range(EPOCH):
        for step, data in enumerate(train_loader, 0):  # gives batch data, normalize x when iterate train_loader
            train_data_x, train_data_y_name, train_data_y_time, encode_y_name = data
            try:
                b_x = train_data_x.view(64, 1, 128, 17)
                output = cnn(b_x)  # cnn output
                # y_name
                #  MSELoss
                loss1 = loss_func(output[0].view(64, 64, 16), train_data_y_name)
                # similarity, words = torch.topk(torch.mv(embedding.weight, output[0][0].clone()), 5)
                loss2 = loss_func(output[1].view(64, 64), train_data_y_time)
                loss = loss1 + loss2

                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients
                # if step % 50 == 0:
                print(step, loss1, loss2, loss)
                print(step, loss1, loss2, loss, file=log_f)
            except:
                print(train_data_x.shape)
            if (step + 1) % 100 == 0:
                get_accuracy(cnn, epoch)
        get_accuracy(cnn, epoch)

        print("保存第 %d 轮结果" % epoch)
        module_name = "module/" + verison + "epoch_" + str(epoch) + ".pickle"
        # with open(module_name, "wb") as f:
        torch.save(cnn, module_name)
print("训练结束...")
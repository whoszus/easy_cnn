#
# todo 1. 需要修改为先各自embedding 之后合并tensor ！

import pandas as pd
from sklearn import preprocessing
import pickle
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
from pytorch_version.NNModule import NetAY

col_names = ["dev_name", "time", "dev_type", "city", "alm_level"]
need_data_changed = False
batch_x = 10
batch_y = 5
LR = 0.001
EPOCH = 3
BATCH_SIZE = 5
load_pickle_data = False

# 声明为全局变量
embedding = nn.Embedding(800, 64)


def load_csv_data(file):
    data = pd.read_csv(file, names=col_names, encoding='utf-8')
    data['time'] = pd.to_datetime(data['time'])
    # todo 考虑是否去重
    data.drop_duplicates(inplace=True)
    return data


# 将时间处理成时间间隔
def time_split(train_data_x):
    print(type(train_data_x))
    c_time = train_data_x['time']
    r_time = []
    for index, value in c_time.iteritems():

        if index % batch_x == 0:
            r_time.append(0)
        else:
            seconds = (c_time[index + 1] - c_time[index]).seconds
            # if seconds > 1000:
            #     print(c_time[index], c_time[index + 1], seconds)
            #     print(index)
            r_time.append(seconds)
    train_data_x['time'] = r_time
    return train_data_x


def embedd(input_data_x, input_dim=800, output_dim=64):
    print("start embedding.....")
    output_x = embedding(input_data_x)
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
#     ['tokyo', 'tokyo', 'paris']
def dataEncode(train_data_X):
    x_les = []
    for name in col_names:
        le = preprocessing.LabelEncoder()
        le.fit(train_data_X[name])
        x_les.append(le)
        train_data_X[name] = le.transform(train_data_X[name])
    # dict
    with open('pickle/les.pickle', 'wb') as feature:
        pickle.dump(x_les, feature, -1)
    return train_data_X


# 重制数据格式为 【batch,5,64】
def data_reshape(train_data_x):
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
    return group_data, group_data_name, group_data_time


def caculateOutdim(input_dim):
    return 50 if input_dim > 50 else (input_dim + 1) / 2


def pickle_loader(input):
    item = pickle.load(open(input, 'rb'))
    return item


def load_data(data_type='train'):
    if data_type == 'train':
        data = load_csv_data("data/data_1.csv")
    else:
        data = load_csv_data("data/data_2.csv")
    data = time_split(data)
    encode_X = dataEncode(data)
    # group data
    encode_X, encode_y_name, encode_y_time = data_reshape(encode_X)
    train_data_X = []
    train_data_y_name = []
    train_data_y_time = []
    for group_data in encode_X:
        train_data_X.append(embedd(group_data))

    for group_data in encode_y_name:
        train_data_y_name.append(embedd(group_data))

    for group_data in encode_y_time:
        train_data_y_time.append(embedd(group_data))
    return train_data_X, train_data_y_name, train_data_y_time


def accuracy_calculate(res, y):
    res = np.array(res.detach())
    y = np.array(y.detach())
    mg = np.intersect1d(res, y)
    return float(len(mg)/len(res))


if __name__ == "__main__":
    if load_pickle_data:
        pickle_train = open('pickle/train_data.pickle', 'rb')
        train_data_X, train_data_y_name, train_data_y_time = pickle.load(pickle_train)
        pickle_test = open('pickle/test_data.pickle', 'rb')
        test_data_X, test_data_y_name, test_data_y_time = pickle.load(pickle_test)
    else:
        train_data_X, train_data_y_name, train_data_y_time = load_data('train')
        test_data_X, test_data_y_name, test_data_y_time = load_data('test')

        # pickle dump data
        with open('pickle/train_data.pickle', 'wb')as f:
            pickle.dump((train_data_X, train_data_y_name, train_data_y_time), f, -1)
        with open('pickle/test_data.pickle', 'wb')as f:
            pickle.dump((test_data_X, test_data_y_name, test_data_y_time), f, -1)

    train_loader = torch.utils.data.DataLoader(dataset=train_data_X, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_data_X, batch_size=BATCH_SIZE, shuffle=False)

    # 开始训练

    cnn = NetAY(batch_x)
    print(cnn)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.MSELoss()  # the target label is not one-hotted

    for epoch in range(EPOCH):
        for step, b_x in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader
            output = cnn(b_x)  # cnn output
            y_name = train_data_y_name[step]
            y_name = y_name.detach()
            y_time = train_data_y_time[step]
            y_time = y_time.detach()
            loss1 = loss_func(output[0], y_name)  # cross entropy loss
            loss2 = loss_func(output[1], y_time)
            loss = loss1 + loss2
            print(step, loss1, loss2, loss)
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward(retain_graph=True)  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            if step % 50 == 0:
                for i, t_x in enumerate(test_loader):
                    test_output = cnn(t_x)
                    accuracy_name = accuracy_calculate(test_output[0], test_data_y_name[i])
                    accuracy_time = accuracy_calculate(test_output[1], test_data_y_time[i])
                    print('Epoch: ', epoch, '| current loss : %.4f' % loss.data.numpy(),
                          '| test accuracy_name: %.2f' % accuracy_name,
                          'accuracy_time:%.2f' % accuracy_time)
                    break

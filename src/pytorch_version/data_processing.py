# coding=utf-8
# -*w- coding utf-8 -*-

# todo 1. 需要修改为先各自embedding 之后合并tensor ！

import pandas as pd
from sklearn import preprocessing
import pickle
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
from NNModule import NetAY
import datetime

col_names = ["dev_name", "time", "dev_type", "city", "alm_level"]
need_data_changed = False
batch_x = 128
batch_y = 64
# LR = 0.001
LR = 0.003
EPOCH = 60
BATCH_SIZE = 64
load_pickle_data = False

# 声明为全局变量
embedding = nn.Embedding(500, batch_y)


# 加载数据
def load_data(data_type='train'):
    if data_type == 'train':
        data = load_csv_data("data/data_2_500w.csv")
    else:
        print("进行测试.....")
        data = load_csv_data("data/test_1_8k.csv")
    # 按时间切分
    data = time_split(data)
    # 转化为标签数据
    encode_x = data_encode(data)
    # group data
    encode_x, encode_y_name, encode_y_time = data_reshape_step(encode_x)
    train_data_x = []
    train_y_name = []
    train_y_time = []
    for group_data in encode_x:
        train_data_x.append(embedd(group_data))

    print("embedding 结束：")

    for group_data in encode_y_name:
        train_y_name.append(embedd(group_data))

    for group_data in encode_y_time:
        train_y_time.append(embedd(group_data))

    return train_data_x, train_y_name, train_y_time, encode_y_name, encode_y_time


# 加载数据 &drop_duplicates
def load_csv_data(file):
    print("开始加载数据..")
    data = pd.read_csv(file, names=col_names, encoding='utf-8')
    data = data.drop_duplicates().dropna().reset_index(drop=True)
    data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S', infer_datetime_format=True, errors="raise")
    print("数据加载完毕，去重完毕，去重后数据量：%d" % len(data))
    return data


# 将时间处理成时间间隔
def time_split(train_data_x):
    print("开始处理时间格式...")
    print(type(train_data_x))
    c_time = train_data_x['time']
    r_time = []

    for index, value in c_time.iteritems():
        try:
            if index % batch_x == 0:
                r_time.append(0)
            else:
                # try:
                # print(index,c_time[index],c_time[index-1])

                seconds = (c_time[index] - c_time[index - 1]).seconds
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
    print("处理时间格式完毕..", train_data_x.head())
    return train_data_x


def embedd(input_data_x, input_dim=800, output_dim=64):
    # print("开始embedding...", embedding)
    # print("start embedding.....")
    output_x = embedding(input_data_x)
    # print("embedding结束..")
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
    print("开始转换数据格式》...")
    x_les = []
    for name in col_names:
        le = preprocessing.LabelEncoder()
        le.fit(train_data_X[name])
        x_les.append(le)
        train_data_X[name] = le.transform(train_data_X[name])
    # dict
    with open('pickle/les.pickle', 'wb') as feature:
        pickle.dump(x_les, feature, -1)
    print("转换数据完毕》。", train_data_X.head(), train_data_X.shape)
    return train_data_X


# 重制数据格式为 【batch,5,64】
def data_reshape(train_data_x):
    print("开始组装数据..")
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
    print("数据组装完毕...", datetime)
    return group_data, group_data_name, group_data_time


# 重制数据格式为 【batch,5,64】
def data_reshape_step(train_data_x, step_i=2):
    if not step_i:
        return data_reshape(train_data_x)

    print("开始组装数据..步长", step_i)
    tmp = []
    group_data = []
    tmp_y = []
    group_data_name = []
    group_data_time = []
    train_data_x = np.array(train_data_x)
    [rows, cols] = train_data_x.shape
    current_i = 1
    i = 1
    while i < rows:
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
            current_i += step_i
            i = current_i
            tmp = []
            tmp_y = []
    print("数据组装完毕...", datetime)
    return group_data, group_data_name, group_data_time


# 暂未使用
def calculate_outdim(input_dim):
    return 50 if input_dim > 50 else (input_dim + 1) / 2


def pickle_loader(input):
    item = pickle.load(open(input, 'rb'))
    return item


# ban
def accuracy_calculate(res, y, train_data_X):
    res = np.array(res.detach())
    y = np.array(y.detach())
    mg = np.intersect1d(res, res)
    # print("")

    return float(len(mg) / len(res))
    # res = res.squeeze().tolist()
    # y = y.squeeze().tolist()
    # return


# todo
def get_accuracy(module):
    if load_pickle_data:
        pickle_test = open('pickle/test_data.pickle', 'rb')
        test_data_x, test_data_y_name, test_data_y_time, e_y_name, e_y_time = pickle.load(pickle_test)
    else:
        print("装载测试数据")
        test_data_x, test_data_y_name, test_data_y_time, e_y_name, e_y_time = load_data('test')
        with open('pickle/test_data.pickle', 'wb')as f:
            pickle.dump((test_data_x, test_data_y_name, test_data_y_time, e_y_name, e_y_time), f, -1)

    # print(module) 开始进行测试
    test_loader = torch.utils.data.DataLoader(dataset=test_data_x, batch_size=BATCH_SIZE, shuffle=False)
    for i, t_x in enumerate(test_loader):
        test_output = module(t_x)
        name_l = test_output[0].chunk(chunks=batch_y, dim=0)
        time_l = test_output[1].chunk(chunks=batch_y, dim=0)
        result_n = []
        result_t = []
        result_n_s = []
        for name in name_l:
            # 这一步操作是将embedding的数据类似翻译回来
            similarity, words = torch.topk(torch.mv(embedding.weight, name.clone().detach().flatten()), 5)
            result_n.append(np.array(words))
            result_n_s.append(similarity.detach().numpy())
        name_acy = get_name_acy(result_n, result_n_s, e_y_name[i])
        for time in time_l:
            similarity, words = torch.topk(torch.mv(embedding.weight, time.clone().detach().flatten()), 5)
            result_n.append(np.array(words))
        time_acy = get_tim_acy(result_t, e_y_time[i])

        print('current epoch :%d ' % epoch, '| test accuracy_name: %.2f' % name_acy, 'accuracy_time:%.2f' % time_acy)


def get_name_acy(m_res, m_res_s, y):
    res = np.array(m_res)
    res_s = np.array(m_res_s)
    y = np.array(y)
    print("预测结果：", res, res_s)
    print("实际结果：", y)
    mg = np.intersect1d(res, y)
    print("交集：", mg)
    return float(len(mg) / len(y))


# todo ; How to measure time accuracy
def get_tim_acy(m_res, y):
    return 1


# main
if __name__ == "__main__":
    print("start learning")
    if load_pickle_data:
        pickle_train = open('pickle/train_data.pickle', 'rb')
        train_data_X, train_data_y_name, train_data_y_time, encode_y_name, encode_y_time = pickle.load(pickle_train)
    else:
        train_data_X, train_data_y_name, train_data_y_time, encode_y_name, encode_y_time = load_data('train')
        # pickle dump data
        with open('pickle/train_data.pickle', 'wb')as f:
            pickle.dump((train_data_X, train_data_y_name, train_data_y_time), f, -1)

    train_loader = torch.utils.data.DataLoader(dataset=train_data_X, batch_size=BATCH_SIZE, shuffle=False)

    # 开始训练
    cnn = NetAY(batch_x, batch_y)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.MSELoss()  # the target label is not one-hotted

    for epoch in range(EPOCH):
        for step, b_x in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader
            print(step, b_x.shape)
            if b_x.shape == torch.Size([batch_y, 256, 5, 64]):
                output = cnn(b_x)  # cnn output
                y_name = train_data_y_name[step]
                y_name = y_name.detach()
                y_time = train_data_y_time[step]
                y_time = y_time.detach()

                #  MSELoss
                loss1 = loss_func(output[0], y_name)
                loss2 = loss_func(output[1], y_time)
                loss = loss1 + loss2

                print(step, loss1, loss2, loss)
                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward(retain_graph=True)  # backpropagation, compute gradients
                optimizer.step()  # apply gradients

            # if step % 500 == 0:
        get_accuracy(cnn)
        print("保存第 %d 轮结果" % epoch)
        module_name = "module/epoch_" + str(epoch) + ".pickle"
        # with open(module_name, "wb") as f:
        torch.save(cnn, module_name)
    print("训练结束...")

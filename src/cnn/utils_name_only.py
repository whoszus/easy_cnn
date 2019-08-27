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
from NNModule import NetAY_name_only
import os
import threading

import copy

# from cachetools import cached, TTLCache

LR = 0.003
EPOCH = 200

col_names = ["city", "dev_name", "dev_type", "time", "alm_level"]
need_data_changed = False
# LR = 0.001
BATCH_SIZE = 32
load_pickle_data = False
c_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
log_f = open("logs/" + c_time + '.log', 'w+')

train_f = 'data/data_train_2_sort_5w.csv'
test_f = "data/data_test_2_sort_5w.csv"
# embedding_time = nn.Embedding(512, 8)
batch_x = 128
batch_y = 64
verison = 'm_1006_5w_'

# GPU = torch.cuda.is_available()
GPU = False

embedding = nn.Embedding(728, 16)

test_pickle_name = 'pickle/test_data.pickle'

device = torch.device("cuda:3" if GPU else "cpu")
device_cpu = torch.device("cpu")
#
train_data_store = 'pickle/train_2_5w.pickle'
test_data_store = 'pickle/test_2_5w.pickle'
module_path = 'module/ERROR.PICKLE'


class MyDataSet(Dataset):
    """ my dataset."""

    # Initialize your data, download, etc.
    def __init__(self):
        # 读取csv文件中的数据
        if not os.path.exists(train_data_store):
            train_data_x, train_data_y_name, encode_y_name = load_data_final()
            df = train_data_x, train_data_y_name, encode_y_name
            with open(train_data_store, 'wb') as f:
                pickle.dump(df, f, -1)
            # data_store = pd.HDFStore(train_data_store)
            # data_store['preprocessed_df'] = df
            # data_store.close()
        else:
            with open(train_data_store, 'rb') as f:
                train_data_x, train_data_y_name, encode_y_name = pickle.load(f)
        # print("load hdfs data ....")
        # data_store = pd.HDFStore(train_data_store)
        # preprocessed_df = data_store['preprocessed_df']
        # train_data_x, train_data_y_name,  encode_y_name = preprocessed_df
        # data_store.close()

        self.train_data_x = train_data_x
        self.train_data_y_name = train_data_y_name
        self.encode_y_name = encode_y_name
        self.len = len(self.train_data_x)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.train_data_x[index], \
               self.train_data_y_name[index], \
               self.encode_y_name[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len


class TestDataSet(Dataset):
    """ my dataset."""

    def __init__(self):
        # 读取csv文件中的数据
        train_data_x, train_data_y_name, encode_y_name = load_data_test()
        with open(test_pickle_name, 'wb') as f:
            pickle.dump((train_data_x, train_data_y_name, encode_y_name), f, -1)

        self.train_data_x = train_data_x
        self.train_data_y_name = train_data_y_name
        self.encode_y_name = encode_y_name
        self.len = len(self.train_data_x)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.train_data_x[index], \
               self.train_data_y_name[index], \
               self.encode_y_name[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len


class TestPickleDataSet():
    """ my dataset."""

    # Initialize your data, download, etc.
    def __init__(self):
        with open(test_pickle_name, 'rb') as f:
            data_pickle = pickle.load(f)
            train_data_x, train_data_y_name, encode_y_name = data_pickle
            self.train_data_x = train_data_x
            self.train_data_y_name = train_data_y_name
            self.encode_y_name = encode_y_name
            self.len = len(self.train_data_x)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.train_data_x[index], \
               self.train_data_y_name[index], \
               self.encode_y_name[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len


class M_Test_data():
    """ my dataset."""

    # Initialize your data, download, etc.
    def __init__(self, m_data):
        train_data_x, train_data_y_name, encode_y_name = m_data
        self.train_data_x = train_data_x
        self.train_data_y_name = train_data_y_name
        self.encode_y_name = encode_y_name
        self.len = len(self.train_data_x)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.train_data_x[index], \
               self.train_data_y_name[index], \
               self.encode_y_name[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len


class DataPrefetch():
    def __init__(self, loader):
        self.loader = iter(loader)
        # self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_input = None
            return
        # with torch.cuda.stream(self.stream):
        #     self.next_data = self.next_data.cuda(non_blocking=True)

    def next(self):
        # torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self.preload()
        return data


# 加载数据
def load_data(batch_x, data_type='train'):
    if data_type == 'train':
        print("开始加载数据.....", file=log_f)
        data = load_csv_data(train_f)
    else:
        print("进行测试.....", file=log_f)
        data = load_csv_data(test_f)
    # 按时间切分
    # data = time_split(data, batch_x)
    # 转化为标签数据
    encode_x = data_encode(data)

    # embedding
    dev_name = encode_x['dev_name']
    dev_name_embedding = embedd(torch.from_numpy(dev_name.values))
    # time_array = np.array(encode_x['time'].values, np.float32, copy=False)
    # todo
    # time_tensor = torch.from_numpy(time_array).view(time_array.size, 1)
    # train_x = torch.cat((dev_name_embedding, time_tensor), 1)
    # group data
    group_data, group_data_name = data_reshape_step(dev_name_embedding, batch_x=batch_x,
                                                    batch_y=batch_y)
    return group_data, group_data_name, reshape_encode_data(dev_name)


# 加载数据 &drop_duplicates
def load_csv_data(file):
    time_start = time.time()
    print("开始加载数据..", file=log_f)
    data = pd.read_csv(file, names=col_names, encoding='utf-8')
    data = data.drop_duplicates().dropna().reset_index(drop=True)
    data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S', infer_datetime_format=True,
                                  errors="raise")
    print("数据加载完毕，去重完毕，去重后数据量：%d" % len(data), "耗时：", time.time() - time_start)
    return data


# 将时间处理成时间间隔
def time_split(train_data_x, batch_x):
    time_start = time.time()
    print("开始处理时间格式...")
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
    print("处理时间格式完毕..", train_data_x.head(), "用时：", time.time() - time_start)
    return train_data_x


def embedd(input_data_x, type='dev_name'):
    if type == 'dev_name':
        output_x = embedding(input_data_x.clone().detach().requires_grad_(True))
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
    print("开始转换数据格式》...")
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
    print("转换数据完毕》。", train_data_X.head(), train_data_X.shape)
    return train_data_X


# 重制数据格式为 【batch,5,64】
def data_reshape(train_data_x, batch_x, batch_y):
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
    print("数据组装完毕...")
    return group_data, group_data_name, group_data_time


def reshape_encode_data(encode_data, step_i=1):
    time_start = time.time()
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
    print("耗时:", time.time() - time_start)
    return group_data


# 重制数据格式为
def data_reshape_step(train_data_x, batch_x, batch_y, step_i=1):
    if not step_i:
        return data_reshape(train_data_x, batch_x, batch_y)

    print("开始组装数据..步长", step_i)
    tmp = []
    group_data = []
    tmp_y = []
    group_data_name = []
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

            group_data_name.append(
                torch.tensor(data_y, dtype=torch.float32))  # 每batch_y条作为一个group 放入group_data_name
            current_i += step_i
            i = current_i
            print("当前装载进度：", current_i)
            tmp = []
            tmp_y = []

    group_data.pop(-1)  # 最后多出来一条没有对应的Y 值
    print("数据组装完毕...", time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    print("数据增加后数量....", len(group_data_name))
    return group_data, group_data_name


# 暂未使用
def calculate_outdim(input_dim):
    return 50 if input_dim > 50 else (input_dim + 1) / 2


def pickle_loader(input):
    item = pickle.load(open(input, 'rb'))
    return item


def acy(module, test_data_x, encode_y_name, step, epoch, best_n):
    test_output = module(test_data_x.view(1, 1, 128, 16))
    name_res = test_output[0].view(64, 16)
    # time_res = test_output[1].view(64)

    result_n = []
    result_n_s = []
    count = 0
    for name in name_res:
        # 这一步操作是将embedding的数据类似翻译回来
        similarity, words = torch.topk(torch.mv(embedding.weight, name.clone()), 3)
        result_n.append(np.array(words))
        result_n_s.append(similarity.detach().numpy())

    name_acy = get_name_acy(result_n, result_n_s, encode_y_name)

    if name_acy > 0.5:
        count += 1
    if step > 500:
        print("只测试500 个数据,最好结果 %d" % (best_n * 100), '%')
        raise Exception
    if step > 100 and best_n < 0.3:
        print("跳过本轮测试")
        raise Exception

    print(step, 'current epoch :%d ' % epoch, '| test accuracy_name: %.2f' % name_acy,
          'acy beyond 50: ', count)
    print(step, 'current epoch :%d ' % epoch, '| test accuracy_name: %.2f' % name_acy,
          'acy beyond 50: ', count, file=log_f)
    return best_n if name_acy < best_n else name_acy


def get_accuracy_tiny(cnnt, epoch, data_test):
    time_start = time.time()
    best_n = 0.00
    best_t = 0.00
    step = 0
    module = copy.deepcopy(cnnt)
    module = module.to(device_cpu)
    test_data_set = M_Test_data(data_test)
    try:
        while step < test_data_set.len:
            test_data_x, test_data_y_name, encode_y_name = test_data_set.__getitem__(step)
            best_n = acy(module, test_data_x, encode_y_name, step, epoch, best_n)
            step += 1
    except Exception as e:
        print(repr(e))
    # torch.save(module, "modules/tmp" + str(best_n) + ".pickle")
    print("test 500 cost time : ", time.time() - time_start, "best:", best_n)


def get_name_acy(m_res, m_res_s, y):
    res = np.array(m_res)
    res_sim = np.array(m_res_s)  # 相似度
    y = np.array(y)
    print("预测结果：", res.flatten(), res_sim.flatten(), file=log_f)
    print("实际结果：", y.flatten(), file=log_f)
    print("预测结果：", res.flatten())
    print("实际结果：", y.flatten())
    mg = np.intersect1d(res, y)
    print("交集：", mg)
    print("交集：", mg, file=log_f)

    if len(mg) > 0:
        # 此部分代码是同个位置，不合理，应为出现就算
        # bc1 = np.bincount(res.flatten())
        # bc2 = np.bincount(y.flatten())
        # # 统计相同元素匹配个数
        # same_count_list = [min(bc1[x], bc2[x]) for x in mg]
        # same_count = sum(same_count_list)
        # print("交集：", mg, file=log_f)
        print("准确/预测总量:", len(mg), len(np.unique(m_res)))

        return float(len(mg) / len(np.unique(y)))
    print("此轮无结果.......", file=log_f)
    return 0.00


# todo ; How to measure time accuracy
def get_tim_acy(m_res, y):
    return 1


def load_data_final():
    print("start learning")
    if load_pickle_data:
        train_data_X, train_data_y_name, encode_y_name, y_time = open('pickle/train_data.pickle',
                                                                      'rb')
    else:
        train_data_X, train_data_y_name, encode_y_name = load_data(batch_x=batch_x)
        # pickle dump data
        # with open('pickle/train_data.pickle', 'wb')as f:
        #     pickle.dump((train_data_X, train_data_y_name, train_data_y_time), f, -1)
    return train_data_X, train_data_y_name, encode_y_name


def load_data_test():
    print("装载测试数据")
    if not os.path.exists(test_data_store):
        data_test_store = load_data(data_type='test', batch_x=batch_x)
        with open(test_data_store, 'wb') as f:
            pickle.dump(data_test_store, f, -1)
        # data_store = pd.HDFStore(train_data_store)
        # data_store['data_test'] = data_test_store
        # data_store.close()
    else:
        with open(test_data_store, 'rb') as f:

            data_test_store = pickle.load(f)
    # print("load hdfs test data ....")
    # data_store = pd.HDFStore(train_data_store)
    # data_test_store = data_store['data_test']
    # data_store.close()
    return data_test_store


def train(cnn, data_test):
    cnn = cnn.to(device)

    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
    # optimizer = torch.optim.SGD(cnn.parameters(), lr=LR)  # optimize all cnn parameters

    loss_func = nn.MSELoss().to(device) if GPU else nn.MSELoss()  # the target label is not one-hotted
    loss_func_name = nn.CrossEntropyLoss().to(device) if GPU else nn.CrossEntropyLoss()
    my_data_set = MyDataSet()
    print("data ready ")
    train_loader = DataLoader(dataset=my_data_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True)
    print("dataloader ready ")
    # prefetcher = DataPrefetch(train_loader)
    # data = prefetcher.next()
    # step = 0
    for epoch in range(EPOCH):
        for step, data in enumerate(train_loader, 0):  # gives batch data, normalize x when iterate train_loader
            # while data is not None:
            # print(step, len(data))
            train_data_x, train_data_y_name, encode_y_name = data
            train_data_x, train_data_y_name, encode_y_name = \
                train_data_x.to(device), train_data_y_name.to(device), encode_y_name.to(device)

            batch_size = len(train_data_x)
            # try:
            b_x = train_data_x.view(batch_size, 1, 128, 16)
            output = cnn(b_x)  # cnn output
            #  MSELoss
            loss1 = loss_func(output[0], train_data_y_name.view(batch_size, -1))
            # similarity, words = torch.topk(torch.mv(embedding.weight, output[0][0].clone()), 5)
            loss = loss1

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            # if step % 50 == 0:
            print(step, loss1, loss)
            # print(step, loss1, loss2, loss, file=log_f)
            # except Exception , err :
            #     print(err)
            #     print(train_data_x.shape)
            # if (step + 1) % 100 == 0:
            thread1 = threading.Thread(target=get_accuracy_tiny, name="准确率线程1",
                                       args=(cnn, epoch, data_test))
            # get_accuracy_tiny(cnn.to(device_cpu), epoch, data_test)
            thread1.start()
            # step += 1
            # data = prefetcher.next()

        # get_accuracy_tiny(cnn.to(device_cpu), epoch, data_test)
        thread1 = threading.Thread(target=get_accuracy_tiny, name="准确率线程2", args=(cnn, epoch, data_test))
        # get_accuracy_tiny(cnn.to(device_cpu), epoch, data_test)
        thread1.start()

        print("保存第 %d 轮结果" % epoch)
        module_name = "module/" + verison + "epoch_" + str(epoch) + ".pickle"
        # with open(module_name, "wb") as f:
        torch.save(cnn, module_name)


if __name__ == '__main__':
    data_test = load_data_test()

    if os.path.exists(module_path):
        print("加载预训练模块")
        cnn = torch.load(module_path)
    else:
        cnn = NetAY_name_only()
    # cnn.share_memory()
    print(cnn)
    train(cnn, data_test)
    # processes = []
    # # 开启多进程
    # for rank in range(num_processes):
    #     p = mp.Process(target=train, args=(cnn, data_test,))
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()
    print("训练结束...")

# coding=utf-8
# -*w- coding utf-8 -*-

import pandas as pd
from sklearn import preprocessing
import pickle
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
import time
from NNModule import NetAY

LR = 0.003
EPOCH = 200


class Utils():
    col_names = ["dev_name", "time", "dev_type", "city", "alm_level"]
    need_data_changed = False
    # LR = 0.001
    EPOCH = 60
    BATCH_SIZE = 64
    load_pickle_data = False
    c_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_f = open("logs/" + c_time + '.log', 'w+')

    train_f = 'data/data_1.csv'
    test_f = "data/test_1_8k.csv"
    embedding = nn.Embedding(728, 16)
    # embedding_time = nn.Embedding(512, 8)
    batch_x = 128
    batch_y = 64

    # 声明为全局变量

    # 加载数据
    def load_data(self, batch_x, data_type='train'):
        if data_type == 'train':
            print("开始加载数据.....", file=self.log_f)
            data = self.load_csv_data(self.train_f)
        else:
            print("进行测试.....", file=self.log_f)
            data = self.load_csv_data(self.test_f)
        # 按时间切分
        data = self.time_split(data, batch_x)
        # 转化为标签数据
        encode_x = self.data_encode(data)

        # embedding
        dev_name = encode_x['dev_name']
        dev_name_embedding = self.embedd(torch.from_numpy(dev_name.values))
        time_array = np.array(encode_x['time'].values, np.float32, copy=False)
        # todo
        time_tensor = torch.from_numpy(time_array).view(time_array.size, 1)
        train_x = torch.cat((dev_name_embedding, time_tensor), 1)
        # group data
        group_data, group_data_name, group_data_time = self.data_reshape_step(train_x, batch_x=self.batch_x,
                                                                              batch_y=self.batch_y)
        return group_data, group_data_name, group_data_time, dev_name

    # 加载数据 &drop_duplicates
    def load_csv_data(self, file):
        print("开始加载数据..", file=self.log_f)
        data = pd.read_csv(file, names=self.col_names, encoding='utf-8')
        data = data.drop_duplicates().dropna().reset_index(drop=True)
        data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S', infer_datetime_format=True,
                                      errors="raise")
        print("数据加载完毕，去重完毕，去重后数据量：%d" % len(data), file=self.log_f)
        return data

    # 将时间处理成时间间隔
    def time_split(self, train_data_x, batch_x):
        print("开始处理时间格式...", file=self.log_f)
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
        print("处理时间格式完毕..", train_data_x.head(), file=self.log_f)
        return train_data_x

    def embedd(self, input_data_x, type='dev_name'):
        if type == 'dev_name':
            output_x = self.embedding(input_data_x)
            # todo save embedding entiy
        else:
            output_x = self.embedding_time(input_data_x)
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
    def data_encode(self, train_data_X):
        print("开始转换数据格式》...", file=self.log_f)
        x_les = []
        for name in self.col_names:
            if name != 'time':
                le = preprocessing.LabelEncoder()
                le.fit(train_data_X[name])
                x_les.append(le)
                train_data_X[name] = le.transform(train_data_X[name])
        # dict
        with open('pickle/les.pickle', 'wb') as feature:
            pickle.dump(x_les, feature, -1)
        print("转换数据完毕》。", train_data_X.head(), train_data_X.shape, file=self.log_f)
        return train_data_X

    # 重制数据格式为 【batch,5,64】
    def data_reshape(self, train_data_x, batch_x, batch_y):
        print("开始组装数据..", file=self.log_f)
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
        print("数据组装完毕...", file=self.log_f)
        return group_data, group_data_name, group_data_time

    # 重制数据格式为 【batch,5,64】
    def data_reshape_step(self, train_data_x, batch_x, batch_y, step_i=12):
        if not step_i:
            return self.data_reshape(train_data_x, self.batch_x, self.batch_y)

        print("开始组装数据..步长", step_i, file=self.log_f)
        tmp = []
        group_data = []
        tmp_y = []
        group_data_name = []
        group_data_time = []
        train_data_x = train_data_x.detach().numpy()
        [rows, cols] = train_data_x.shape
        current_i = 1
        i = 1
        while i < rows:
            tmp.append(train_data_x[i])
            tmp_y.append(train_data_x[i])
            i += 1
            if len(tmp) % batch_x == 0:
                group_data.append(torch.tensor(tmp))
                tmp = []
            if len(tmp) % (batch_x + batch_y) == 0:
                data_y = tmp_y[batch_y * -1:]  # 将倒数batch_y 条放入data_y
                data_y = np.array(data_y)

                data_y_name = data_y[:, 0:16]  # 第一列放入data_y_name

                data_y_time = data_y[:, 16]  # 第二列放入data_y_time
                group_data_name.append(
                    torch.tensor(data_y_name, dtype=torch.long))  # 每batch_y条作为一个group 放入group_data_name
                group_data_time.append(torch.tensor(data_y_time, dtype=torch.long))
                current_i += step_i
                i = current_i
                print("当前装载进度：", current_i, file=self.log_f)
                tmp = []
                tmp_y = []
        print("数据组装完毕...", time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()), file=self.log_f)
        print("数据增加后数量....", len(group_data_name), file=self.log_f)
        return group_data, group_data_name, group_data_time

    # 暂未使用
    def calculate_outdim(self, input_dim):
        return 50 if input_dim > 50 else (input_dim + 1) / 2

    def pickle_loader(self, input):
        item = pickle.load(open(input, 'rb'))
        return item

    # todo
    def get_accuracy(self, module, epoch):
        if self.load_pickle_data:
            pickle_test = open('pickle/test_data.pickle', 'rb')
            test_data_x, test_data_y_name, test_data_y_time, e_y_name = pickle.load(pickle_test)
        else:
            print("装载测试数据")
            test_data_x, test_data_y_name, test_data_y_time, e_y_name = self.load_data(data_type='test',
                                                                                       batch_x=self.batch_x)
            with open('pickle/test_data.pickle', 'wb')as f:
                pickle.dump((test_data_x, test_data_y_name, test_data_y_time, e_y_name), f, -1)

        # print(module) 开始进行测试
        test_loader = torch.utils.data.DataLoader(dataset=test_data_x, batch_size=self.BATCH_SIZE, shuffle=False)
        for i, t_x in enumerate(test_loader):
            test_output = module(t_x)
            name_l = test_output[0].chunk(chunks=self.batch_y, dim=0)
            time_l = test_output[1].chunk(chunks=self.batch_y, dim=0)
            result_n = []
            result_t = []
            result_n_s = []
            for name in name_l:
                # 这一步操作是将embedding的数据类似翻译回来
                similarity, words = torch.topk(torch.mv(self.embedding.weight, name.clone().detach().flatten()), 5)
                result_n.append(np.array(words))
                result_n_s.append(similarity.detach().numpy())
            name_acy = self.get_name_acy(result_n, result_n_s, e_y_name[i])
            for time in time_l:
                similarity, words = torch.topk(torch.mv(self.embedding.weight, time.clone().detach().flatten()), 5)
                result_n.append(np.array(words))
            time_acy = self.get_tim_acy(result_t, test_data_y_time[i])

            print('current epoch :%d ' % epoch, '| test accuracy_name: %.2f' % name_acy,
                  'accuracy_time:%.2f' % time_acy)

    def get_name_acy(self, m_res, m_res_s, y):
        res = np.array(m_res)
        res_s = np.array(m_res_s)
        y = np.array(y)
        print("预测结果：", res, res_s, file=self.log_f)
        print("实际结果：", y, file=self.log_f)
        mg = np.intersect1d(res, y)
        print("交集：", mg, file=self.log_f)
        return float(len(mg) / len(y))

    # todo ; How to measure time accuracy
    def get_tim_acy(self, m_res, y):
        return 1

    def load_data_final(self):
        print("start learning")
        if self.load_pickle_data:
            train_data_X, train_data_y_name, train_data_y_time, encode_y_name, y_time = open(
                'pickle/train_data.pickle', 'rb')
        else:
            train_data_X, train_data_y_name, train_data_y_time, encode_y_name = self.load_data(batch_x=self.batch_x)
            # pickle dump data
            # with open('pickle/train_data.pickle', 'wb')as f:
            #     pickle.dump((train_data_X, train_data_y_name, train_data_y_time), f, -1)
        return train_data_X, train_data_y_name, train_data_y_time, encode_y_name


if __name__ == '__main__':
    util = Utils()
    train_data_X, train_data_y_name, train_data_y_time, encode_y_name = util.load_data_final()
    train_loader = torch.utils.data.DataLoader(dataset=train_data_X, batch_size=64, shuffle=False)

    # 开始训练
    cnn = NetAY(128, 64)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.MSELoss()  # the target label is not one-hotted

    for epoch in range(EPOCH):
        for step, b_x in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader
            if b_x.shape[0] == 64:
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
        util.get_accuracy(cnn, epoch)
        print("保存第 %d 轮结果" % epoch)
        module_name = "module/epoch_" + str(epoch) + ".pickle"
        # with open(module_name, "wb") as f:
        torch.save(cnn, module_name)
    print("训练结束...")

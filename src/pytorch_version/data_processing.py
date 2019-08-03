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
batch_x = 100
batch_y = 80
LR = 0.001
EPOCH = 3
BATCH_SIZE = batch_x
train_rate = 0.7


def load_data():
    data = pd.read_csv("data_1.csv", names=col_names, encoding='utf-8')
    # data ['time'] = data['time'].map(lambda x: datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
    data['time'] = pd.to_datetime(data['time'])
    # data.sort_values('time', inplace=True, ascending=True)
    # 去重
    data.drop_duplicates(inplace=True)
    return data


# 将时间处理成时间间隔
def time_split(train_data_x):
    # if ~need_data_changed:
    #     return train_data_x
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


def embedd(input_data_x, input_data_y, input_dim=800, output_dim=64):
    embedding = nn.Embedding(input_dim, output_dim)
    input_x = torch.LongTensor(input_data_x)
    input_y = torch.LongTensor(input_data_y)
    print("start embedding.....")

    output_x = embedding(input_x)
    output_y = embedding(input_y)

    with open("embedded_data.pickle", "wb") as f:
        pickle.dump((output_x, output_y), f, -1)
    return output_x, output_y


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
    with open('les.pickle', 'wb') as feature:
        pickle.dump(x_les, feature, -1)
    return train_data_X


def caculateOutdim(input_dim):
    return 50 if input_dim > 50 else (input_dim + 1) / 2


if __name__ == "__main__":
    data = load_data()
    data = time_split(data)
    encode_X = dataEncode(data)
    encode_X = np.array(encode_X)
    # for c_name in col_names:
    #     data_unique = pd.unique(encode_X[c_name])
    #     out_dim = caculateOutdim(data_unique)
    encode_y = np.delete(encode_X, [2, 3, 4], axis=1)
    embedd_x, encode_y = embedd(encode_X, encode_y)
    print(embedd_x.shape, encode_y.shape)

    # print(encode_X.head())
    # print("\nNum of data: ", len(data), "\n")  # 1728
    # # view data values
    # for name in data.keys():
    #     print(name, pd.unique(data[name]))
    # print("\n", encode_X.head(2))
    # encode_X.to_csv("alarm_onehot.csv", index=False)
    cnn = NetAY()
    print(cnn)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
    train_loader = Data.DataLoader(dataset=embedd_x, batch_size=BATCH_SIZE)

    for epoch in range(EPOCH):
        for step, b_x in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader
            print(type(b_x),b_x.shape)
            output = cnn(b_x)  # cnn output
            # loss = loss_func(output, b_y)  # cross entropy loss
            # optimizer.zero_grad()  # clear gradients for this training step
            # loss.backward()  # backpropagation, compute gradients
            # optimizer.step()  # apply gradients
            #
            # if step % 50 == 0:
            #     test_output, last_layer = cnn(test_x)
            #     pred_y = torch.max(test_output, 1)[1].data.numpy()
            #     accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            #     print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

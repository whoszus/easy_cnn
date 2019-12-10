import timeit

import pandas as pd
import torch
from sklearn import preprocessing
import os
from dataset import M_Test_data
import torch.utils.data
import numpy as np
from collections import Counter

col_names = ["dev_name", "time"]

# torch_save_path = 'data/csv/data_train_2_sort.torch'
file_path = 'data/csv/data_train_2_sort.csv'
save_pt_path = 'data/pt/'
dict_path = 'data/dict/'
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


# 对网元进行编码
def data_encode_sort(train_data_x, dev_name_dict):
    dev_name = train_data_x['dev_name'].map(dev_name_dict)
    train_data_x['dev_name'] = dev_name
    return train_data_x


# 获取设备字典
def get_dict(train_data_x, start, end):
    dict_path_s = dict_path + start + '#' + end + '.pt'
    dict_path_s = dict_path_s.replace(" ", "#").replace(":", "-")
    if os.path.exists(dict_path_s):
        return torch.load(dict_path_s)
    count_set = pd.unique(pd.array(train_data_x['dev_name']))
    dev_name_dict = {w: index + 1 for index, w in enumerate(count_set)}
    torch.save(dev_name_dict, dict_path_s)
    return dev_name_dict


# 加载数据 &drop_duplicates
def load_csv_data(torch_save_path):
    if os.path.exists(torch_save_path):
        print("数据已结构化..读取数据中..")
        return torch.load(torch_save_path)['data_all']
    print("开始加载数据..")
    data = pd.read_csv(file_path, names=col_names, encoding='utf-8')
    data = data.drop_duplicates().dropna().reset_index(drop=True)
    data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S', infer_datetime_format=True, errors="raise")
    torch_save_data = {
        'data_all': data
    }
    torch.save(torch_save_data, torch_save_path)
    print("数据加载完毕，去重完毕，去重后数据量：%d" % len(data))
    return data


def load_data_by_num(data, start, num):
    return data[data['time'] >= start][num]


def load_data_by_date(df, start, end):
    # return data[start <= data['time'] <= end]
    mask = (df['time'] >= start) & (df['time'] <= end)
    return df.loc[mask]


# 加载数据
def load_data(torch_save_path, start, end, num=0):
    print("开始加载数据.....")
    data_load = torch.load(torch_save_path)['data_all']
    start_time = pd.to_datetime(start, format='%Y-%m-%d %H:%M:%S')
    end_time = pd.to_datetime(end, format='%Y-%m-%d %H:%M:%S')
    if num > 0:
        data_load = load_data_by_num(data_load, start_time, num)
    else:
        data_load = load_data_by_date(data_load, start_time, end_time)
    dict_name = get_dict(data_load, start, end)

    df = pd.DataFrame.from_dict(dict_name, orient="index")
    df.to_csv('data/csv/dic.csv')
    data_len = len(data_load['dev_name'])
    data_load = data_encode_sort(data_load, dict_name)
    print("数据量：",len(data_load))
    data_count = data_load['dev_name'].value_counts().to_frame()
    v_size = len(data_count)
    df = pd.concat([df, data_count])
    df.to_csv('data/csv/dic_count.csv')

    data_train = data_load[:int(train_rate * data_len)]

    data_val = data_load[int((1 - train_rate) * data_len) * -1:]

    data_val_ofpa = data_val['dev_name'].value_counts(ascending=True).head(int(v_size*0.05))
    data_val_ofpa = list(data_val_ofpa.to_frame().index)
    # data_val_ofpa = data_val_ofpa.apply()

    data_voc = len(pd.unique(pd.array(data_train['dev_name']))) + len(pd.unique(pd.array(data_val['dev_name'])))

    data = {
        'train_data': data_train,
        'val_data': data_val,
        'voc': data_voc,
        'data_val_ofpa': data_val_ofpa
    }
    path = save_pt_path + start + '#' + end + '.pt'
    path = path.replace(" ", "#").replace(":", "-")
    torch.save(data, path)
    return data


def split_data_set(train_data_set, batch_x, batch_y, device, step_i=12):
    current_i = 1
    step = 0
    tmp = []
    group_data = []
    group_data_y = []
    train_data_set = train_data_set.tolist()
    while step < len(train_data_set):
        tmp.append(train_data_set[step])
        step += 1
        if len(tmp) == batch_x:
            # print("组装中：", step)
            group_data.append(np.array(tmp))
        if len(tmp) == (batch_y + batch_x):
            tmp_y = tmp[batch_y * -1:]
            group_data_y.append(np.array(tmp_y))
            tmp = []
            current_i = current_i + step_i
            step = current_i
    group_data.pop(-1)

    print("data_length", len(group_data))
    return torch.tensor(group_data).to(device), torch.tensor(group_data_y).to(device)


# 将时间处理成时间间隔
def time_split_group(train_data_time, batch_x, batch_y, device, step_i=12):
    tmp = []
    current_i = 1
    group_data = []
    group_data_y = []
    step = 0
    first_time = train_data_time[0]
    get_sec = lambda x, y: (x - y).seconds if x > y else (y - x).seconds

    while step < len(train_data_time):
        if step == 0:
            tmp.append(0)
            step += 1
            continue
        seconds = get_sec(train_data_time[step], first_time)
        tmp.append(seconds)
        step += 1
        if len(tmp) == batch_x:
            group_data.append(tmp)
            first_time = train_data_time[step - 1]

        if len(tmp) == (batch_y + batch_x):
            group_data_y.append(np.array(tmp[batch_y * -1:]))
            tmp = []
            current_i = current_i + step_i
            step = current_i
            first_time = train_data_time[step - 1]
            print("时间处理中...", step)
    group_data.pop(-1)
    return torch.tensor(group_data).to(device), torch.tensor(group_data_y).to(device)


# 将时间处理成时间间隔
def time_split(train_data_time):
    c_time = train_data_time
    r_time = []
    for index, value in c_time.iteritems():
        if index == 0:
            r_time.append(5)
        else:
            get_sec = lambda x, y: (x - y).seconds if x > y else (y - x).seconds
            seconds = get_sec(c_time[index], c_time[index - 1])
            # seconds = seconds if seconds > 1 else 5
            r_time.append(seconds + 5)
    return r_time


def get_data_loader(opt, device):
    if not os.path.exists(opt.data_all):
        print("初始化数据....")
        load_csv_data(opt.data_all)

    dataset_path = 'data/data_set/' + opt.start_time + '#' + opt.end_time + '.pt'
    dataset_path = dataset_path.replace(" ", "#").replace(":", "-")
    if os.path.exists(dataset_path):
        data_torch_loaded = torch.load(dataset_path)
        data_loader = data_torch_loaded['train']
        data_loader_val = data_torch_loaded['val']
        voc_name = data_torch_loaded['voc']
        data_val_ofpa = data_torch_loaded['data_val_ofpa']
    else:
        tmp_data_path = save_pt_path + opt.start_time + '#' + opt.end_time + '.pt'
        tmp_data_path = tmp_data_path.replace(" ", "#").replace(":", "-")

        if os.path.exists(tmp_data_path):
            data_train = torch.load(tmp_data_path)['train_data']['dev_name']
            data_val = torch.load(tmp_data_path)['val_data']['dev_name']
        else:
            data_tmp = load_data(opt.data_all, opt.start_time, opt.end_time)
            data_train = data_tmp['train_data']['dev_name']
            data_val = data_tmp['val_data']['dev_name']

        m_data = split_data_set(data_train, opt.batch_x, opt.batch_y, device)
        data_set_train = M_Test_data(m_data)
        data_loader = torch.utils.data.DataLoader(data_set_train, batch_size=opt.batch_size, shuffle=True,
                                                  drop_last=True)

        m_data_val = split_data_set(data_val, opt.batch_x, opt.batch_y, device)
        data_set_val = M_Test_data(m_data_val)
        data_loader_val = torch.utils.data.DataLoader(data_set_val, batch_size=opt.batch_size, shuffle=True,
                                                      drop_last=True)

        voc_name = torch.load(tmp_data_path)['voc']
        data_val_ofpa = torch.load(tmp_data_path)['data_val_ofpa']

        data_loader_p = {
            'train': data_loader,
            'val': data_loader_val,
            'voc': voc_name,
            'data_val_ofpa': data_val_ofpa
        }
        torch.save(data_loader_p, dataset_path)
    return data_loader, data_loader_val, voc_name, data_val_ofpa


def get_time_vac(opt):
    train_data = torch.load(opt.data_all)['train_data']['time']
    train_data_y = time_split(train_data)
    train_data_y = np.array(train_data_y)
    size = np.unique(train_data_y).size
    return size


if __name__ == '__main__':
    start_time_str = '2018-07-01'
    end_time_str = '2018-08-01'
    load_data('data/csv/data_train_2_sort.torch', start_time_str, end_time_str)

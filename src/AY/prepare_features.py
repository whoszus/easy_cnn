import pickle
from datetime import datetime
from sklearn import preprocessing
import numpy as np

batch_x = 100
batch_y = 30

with open('train_data.pickle', 'rb') as feature:
    train_data = pickle.load(feature)
    num_records = len(train_data)
# with open('test_data.pickle', 'rb') as t_f:
#     test_data = pickle.load(t_f)
#     len_test_data = len(test_data)


def feature_list(alarms, lat):
    if lat != '':
        alarm_time = datetime.strptime(alarms['time'], '%Y-%m-%d %H:%M:%S')
        last_alarm_time_dt = datetime.strptime(lat, '%Y-%m-%d %H:%M:%S')
        time = (last_alarm_time_dt - alarm_time).seconds
        if time > 10:
            print(last_alarm_time,alarm_time,time)
    else :
        time = 0
    alarm_level = int(alarms['alm_level'])
    city = int(alarms['city'])
    dev_name = int(alarms['dev_name'])
    dev_type = int(alarms['dev_type'])
    return [dev_name, dev_type, city, time, alarm_level]


# 构建label 也需要E
def label_list(alarms, lat):
    alarm_time = datetime.strptime(alarms['time'], '%Y-%m-%d %H:%M:%S')
    last_alarm_time_dt = datetime.strptime(lat, '%Y-%m-%d %H:%M:%S')
    time = (alarm_time - last_alarm_time_dt).seconds
    dev_name = alarms['dev_name']
    return [dev_name, time]


train_data_X = []
train_data_x_tmp = []
train_data_y = []
train_data_y_tmp = []
tmp = []
last_alarm_time = ''
data_index = 1
all_data = []
# 构建参数
for record in train_data:
    feature = feature_list(record, last_alarm_time)
    train_data_x_tmp.append(feature)
    tmp.append(feature)
    all_data.append(feature)
    if data_index % batch_x == 0:
        train_data_X.append(np.array(train_data_x_tmp))
        train_data_x_tmp = []
        last_alarm_time = ''
    if data_index % (batch_x + batch_y) == 0:
        train_data_y_tmp = tmp[batch_y*-1:]
        train_data_y_tmp = np.delete(train_data_y_tmp, [2, 3, 4], axis=1)
        train_data_y.append(train_data_y_tmp)
        tmp =[]
    last_alarm_time = record['time']
    data_index += 1
data_index = 0

full_X = all_data
full_X = np.array(full_X)
all_data = np.array(all_data)
les = []
for i in range(all_data.shape[1]):
    le = preprocessing.LabelEncoder()
    le.fit(full_X[:, i])
    les.append(le)
    all_data[:, i] = le.transform(all_data[:, i])

with open('les.pickle', 'wb') as feature:
    pickle.dump(les, feature, -1)

train_data_y = np.array(train_data_y)
with open('feature_train_data.pickle', 'wb') as feature:
    pickle.dump((train_data_X, train_data_y), feature, -1)
    print(train_data_X[0], train_data_y[0])

import pickle
from datetime import datetime
from sklearn import preprocessing
import numpy as np


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
    city = alarms['city']
    dev_name = alarms['dev_name']
    dev_type = alarms['dev_type']
    return [dev_name, dev_type, city, time, alarm_level]





# func start
train_data_X = []
train_data_x_tmp = []
train_data_y = []
tmp = []
last_alarm_time = ''
data_index = 1
batch_x = 20
batch_y = 5

with open('train_data.pickle', 'rb') as feature:
    train_data = pickle.load(feature)
    num_records = len(train_data)

print("开始处理数据"+str(datetime))
# 构建参数
for record in train_data:
    feature = feature_list(record, last_alarm_time)
    train_data_X.append(feature)
    tmp.append(feature)
    if data_index % batch_x == 0:
        last_alarm_time = ''
    if data_index % (batch_x + batch_y) == 0:
        train_data_y += tmp[batch_y * -1:]
        tmp = []
    last_alarm_time = record['time']
    data_index += 1
data_index = 0

train_data_X = np.array(train_data_X)
x_les = []

"""
    >>> le = preprocessing.LabelEncoder()
    >>> le.fit(["paris", "paris", "tokyo", "amsterdam"])
    LabelEncoder()
    >>> list(le.classes_)
    ['amsterdam', 'paris', 'tokyo']
    >>> le.transform(["tokyo", "tokyo", "paris"]) #doctest: +ELLIPSIS
    array([2, 2, 1]...)
    >>> list(le.inverse_transform([2, 2, 1]))
    ['tokyo', 'tokyo', 'paris']

"""

for i in range(train_data_X.shape[1]):
    le = preprocessing.LabelEncoder()
    le.fit(train_data_X[:, i])
    x_les.append(le)
    train_data_X[:, i] = le.transform(train_data_X[:, i])

with open('x_les.pickle', 'wb') as f:
    pickle.dump(x_les, f, -1)

train_data_y = np.array(train_data_y)

#[dev_name, dev_type, city, time, alarm_level]
train_data_y = np.delete(train_data_y, [1, 2, 4], axis=1)
with open('feature_train_data.pickle_X_size'+str(batch_x)+'_Y_size_'+str(batch_y), 'wb') as f:
    pickle.dump((train_data_X, train_data_y), f, -1)
    print(train_data_X[-1], train_data_y[0])


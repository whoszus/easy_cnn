import pickle
from datetime import datetime
from sklearn import preprocessing
import numpy as np

with open('train_data.pickle', 'rb') as f:
    train_data = pickle.load(f)
    num_records = len(train_data)
with open('test_data.pickle', 'rb') as t_f:
    test_data = pickle.load(t_f)
    len_test_data = len(test_data)


def feature_list(record,last_alarm_time):
    alarm_time = datetime.strptime(record['time'], '%Y-%m-%d %H:%M:%S')
    last_alarm_time_dt = datetime.strptime(last_alarm_time, '%Y-%m-%d %H:%M:%S')
    time = (alarm_time - last_alarm_time_dt).seconds
    alarm_level = int(record['alarm_level'])


    return [city,
            store_index,
            day_of_week,
            promo,
            year,
            month,
            day,
            store_data[store_index - 1]['State']
            ]


train_data_X = []
train_data_y = []
last_alarm_time = ''

# 构建参数
for record in train_data:
    fl = feature_list(record, last_alarm_time)
    train_data_X.append(fl)
    last_alarm_time = record['time']
for record in test_data:
    fl = feature_list(record, last_alarm_time)
    train_data_y.append(fl)
    last_alarm_time = record['time']

print(min(train_data_y), max(train_data_y))



full_X = train_data_X
full_X = np.array(full_X)
train_data_X = np.array(train_data_X)
les = []
for i in range(train_data_X.shape[1]):
    le = preprocessing.LabelEncoder()
    le.fit(full_X[:, i])
    les.append(le)
    train_data_X[:, i] = le.transform(train_data_X[:, i])

with open('les.pickle', 'wb') as f:
    pickle.dump(les, f, -1)

train_data_X = train_data_X.astype(int)
train_data_y = np.array(train_data_y)

with open('feature_train_data.pickle', 'wb') as f:
    pickle.dump((train_data_X, train_data_y), f, -1)
    print(train_data_X[0], train_data_y[0])

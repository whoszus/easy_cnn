import pickle
import csv


def csv2dicts(csv_file):
    data = []
    keys = []
    for row_index, row in enumerate(csv_file):
        if row_index == 0:
            keys = row
            print(row)
            continue
        data.append({key: value for key, value in zip(keys, row)})
    return data


# def set_nan_as_string(data, replace_str='0'):
#     for i, x in enumerate(data):
#         for key, value in x.items():
#             if value == '':
#                 x[key] = replace_str
#         data[i] = x


train_data = "data_train.csv"
test_data = "data_test.csv"

with open(train_data) as csvfile, open(test_data) as test_data_csv:
    data = csv.reader(csvfile)
    t_data = csv.reader(test_data_csv)
    with open('train_data.pickle', 'wb') as f:
        data = csv2dicts(data)
        data = data[::-1]
        pickle.dump(data, f, -1)
        print(data[:3])
    with open('test_data.pickle','wb') as t_f:
        t_data = csv2dicts(t_data)
        data = data[::-1]
        pickle.dump(data, t_f, -1)
        print(t_data[:3])

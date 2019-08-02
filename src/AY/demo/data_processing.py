import pandas as pd
from sklearn import preprocessing
import pickle
import torch
import torch.nn as nn

col_names = ["dev_name", "dev_type", "city", "alm_level"]
def load_data():
    data = pd.read_csv("data_1.csv", names=col_names,encoding='GBK')
    return data

def embedd(input_data, input_dim, output_dim):
    embedding = nn.Embedding(input_dim, output_dim)
    input = torch.LongTensor(input_data)
    return embedding(input)

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


def caculateOutdim(input_dim) :
    return 50 if input_dim>50 else (input_dim+1)/2

if __name__ == "__main__":
    data = load_data()
    encode_X = dataEncode(data)
    for c_name in col_names:
        data_unique = pd.unique(encode_X[c_name])
        out_dim = caculateOutdim(data_unique)


    print(encode_X.head())
    print("\nNum of data: ", len(data), "\n")  # 1728
    # view data values
    for name in data.keys():
        print(name, pd.unique(data[name]))
    print("\n", encode_X.head(2))
    # encode_X.to_csv("alarm_onehot.csv", index=False)
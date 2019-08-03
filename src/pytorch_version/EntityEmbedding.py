import torch
import torch.nn as nn

torch.manual_seed(1)


def embedd(input_data):
    embedding = nn.Embedding(10, 3)
    input = torch.LongTensor(input_data)
    return embedding(input)




data = [[1,2,4,5],[4,3,2,9]]
m = embedd(data)
print(m)
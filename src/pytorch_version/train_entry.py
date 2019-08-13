# coding=utf-8
# -*w- coding utf-8 -*-

import pickle
import torch
import torch.nn as nn
import torch.utils.data as Data
from NNModule import NetAY
from .utils import Utils

batch_x = 128
batch_y = 64
LR = 0.003
EPOCH = 200
# main
if __name__ == "__main__":

    train_data_X, train_data_y_name, train_data_y_time, encode_y_name, encode_y_time = Utils.load_data_final()
    train_loader = torch.utils.data.DataLoader(dataset=train_data_X, batch_size=64, shuffle=False)

    # 开始训练
    cnn = NetAY(batch_x, batch_y)
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
        get_accuracy(cnn)
        print("保存第 %d 轮结果" % epoch)
        module_name = "module/epoch_" + str(epoch) + ".pickle"
        # with open(module_name, "wb") as f:
        torch.save(cnn, module_name)
    print("训练结束...")

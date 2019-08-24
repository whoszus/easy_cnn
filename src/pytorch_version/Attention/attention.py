import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import math
import os
import time
from NSModule import Encoder, EncoderLayer, SelfAttention, PositionwiseFeedforward, \
    DecoderLayer, Seq2Seq, Decoder, NoamOpt
import pickle
from DATA_SET import M_Test_data
import numpy as np


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0
    data_loader_train = DataLoader(dataset=iterator, batch_size=12, shuffle=True, pin_memory=True, drop_last=True)
    for i, batch in enumerate(data_loader_train):
        src = batch[0]
        trg = batch[1]
        if trg.shape[0] == 12:
            optimizer.optimizer.zero_grad()

            output = model(src, trg)

            # output = [batch size, trg sent len - 1, output dim]
            # trg = [batch size, trg sent len]

            output = output.contiguous().view(-1, output.shape[-1])
            # trg = trg[:, 1:].contiguous().view(-1)

            # output = [batch size * trg sent len - 1, output dim]
            # trg = [batch size * trg sent len - 1]
            trg = trg.view(-1)
            loss = criterion(output, trg)
            print("实际结果", trg)
            print(i, loss)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()

            epoch_loss += loss.item()
        else:
            print(trg.shape)

    return epoch_loss / len(iterator)


def evaluate(model, data_set_evaluate, criterion):
    model.eval()
    epoch_loss = 0
    # data_loader_evaluate = DataLoader(dataset=data_set_evaluate, batch_size=12, shuffle=True, pin_memory=True, drop_last=True)
    with torch.no_grad():
        for i, batch in enumerate(data_set_evaluate):
            src = batch[0]
            trg = batch[1]
            if trg.shape[0] == 12:
                output = model(src, trg)

                # output = [batch size, trg sent len - 1, output dim]
                # trg = [batch size, trg sent len]

                output = output.contiguous().view(-1, output.shape[-1])
                # trg = trg[:, 1:].contiguous().view(-1)

                # output = [batch size * trg sent len - 1, output dim]
                # trg = [batch size * trg sent len - 1]

                loss = criterion(output, trg.view(-1))
                print("校验loss", loss)
                epoch_loss += loss.item()
    return epoch_loss / len(data_set_evaluate)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def split_data_set(train_data_set, batch_x, batch_y, step_i=32):
    current_i = 1
    step = 0
    tmp = []
    group_data = []
    group_data_y = []
    while step < len(train_data_set):
        tmp.append(train_data_set[step])
        step += 1
        if len(tmp) % batch_x == 0:
            print("组装中：", step)
            group_data.append(torch.tensor(np.array(tmp), dtype=torch.long))
        if len(tmp) % (batch_y + batch_x) == 0:
            group_data_y.append(torch.tensor(np.array(tmp[batch_y * -1:]), dtype=torch.long))
            tmp = []
            current_i = current_i + step_i
            step = current_i
    group_data.pop(-1)
    return group_data, group_data_y


def train_module():
    optimizer = NoamOpt(hid_dim, 1, 2000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    criterion = nn.CrossEntropyLoss()
    best_valid_loss = float('inf')
    if not os.path.isdir(f'{SAVE_DIR}'):
        os.makedirs(f'{SAVE_DIR}')
    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss = train(model, data_set_train, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, data_set_train, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

        print(
            f'| Epoch: {epoch + 1:03} | Time: {epoch_mins}m {epoch_secs}s| Train Loss: {train_loss:.3f} | '
            f'Train PPL: {math.exp(train_loss):7.3f} |'
            f' Val.Loss: {valid_loss: .3f} |'
            f' Val.PPL: {math.exp(valid_loss): 7.3f} | ')


SAVE_DIR = 'models'
SEED = 1
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128
data_group_save_path = os.path.join('pickle', 'group_data.pickle')
with open('../pickle/name_pickle', 'rb') as f:
    train_x = pickle.load(f)
m_data = split_data_set(train_x, 64, 32)
data_set_train = M_Test_data(m_data)
input_dim = len(train_x)
hid_dim = 512
n_layers = 6
n_heads = 8
pf_dim = 2048
dropout = 0.1
enc = Encoder(input_dim, hid_dim, n_layers, n_heads, pf_dim, EncoderLayer, SelfAttention, PositionwiseFeedforward,
              dropout, device)
output_dim = input_dim
dec = Decoder(output_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward,
              dropout, device)
model = Seq2Seq(enc, dec, 0, device).to(device)
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

N_EPOCHS = 10
CLIP = 1
train_module()

MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'transformer-seq2seq.pt')
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
print(f'The model has {count_parameters(model):,} trainable parameters')

data_loader = DataLoader(data_set_train, batch_size=1, shuffle=True, pin_memory=True, drop_last=True)
criterion = nn.CrossEntropyLoss()

test_loss = evaluate(model, data_loader, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

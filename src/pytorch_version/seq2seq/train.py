'''
This script handling the training process.
'''

import argparse
import math
import time

from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import transformer.Constants as Constants
from dataset import M_Test_data, paired_collate_fn
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim
import numpy as np
import os

device = torch.device('cuda0' if torch.cuda.is_available() else 'cpu')


def cal_performance(pred, gold, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(Constants.PAD)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct


def cal_loss(pred, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(Constants.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='sum')

    return loss


def train_epoch(model, training_data, optimizer, device, smoothing):
    ''' Epoch operation in training phase'''

    model.train()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):
        # prepare data
        src_seq, tgt_seq = map(lambda x: x.to(device), batch)
        #
        src_pos = torch.tensor(get_position(src_seq.shape[1])).to(device)
        tgt_pos = torch.tensor(get_position(tgt_seq.shape[1])).to(device)
        # gold = tgt_seq[:, 1:]

        # forward
        optimizer.zero_grad()
        pred = model(src_seq, src_pos, tgt_seq, tgt_pos)

        # backward
        loss, n_correct = cal_performance(pred, tgt_seq, smoothing=smoothing)
        loss.backward()

        # update parameters
        optimizer.step_and_update_lr()

        # note keeping
        total_loss += loss.item()

        non_pad_mask = tgt_seq.ne(Constants.PAD)
        n_word = non_pad_mask.sum().item()
        n_word_total += n_word
        n_word_correct += n_correct

    loss_per_word = total_loss / n_word_total
    accuracy = n_word_correct / n_word_total
    return loss_per_word, accuracy


def get_position(ranges):
    pos = []
    for i in range(ranges):
        pos.append(i)
    return pos


def eval_epoch(model, validation_data, device):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    with torch.no_grad():
        for batch in tqdm(
                validation_data, mininterval=2,
                desc='  - (Validation) ', leave=False):
            # prepare data
            src_seq, tgt_seq = map(lambda x: x.to(device), batch)
            # gold = tgt_seq[:, 1:]
            src_pos = torch.tensor(get_position(src_seq.shape[1])).to(device)
            tgt_pos = torch.tensor(get_position(tgt_seq.shape[1])).to(device)
            # forward
            pred = model(src_seq, src_pos, tgt_seq, tgt_pos)
            loss, n_correct = cal_performance(pred, tgt_seq, smoothing=False)

            # note keeping
            total_loss += loss.item()

            non_pad_mask = tgt_seq.ne(Constants.PAD)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct

    loss_per_word = total_loss / n_word_total
    accuracy = n_word_correct / n_word_total
    return loss_per_word, accuracy


def train(model, training_data, validation_data, optimizer, device, opt):
    ''' Start training '''

    log_train_file = None
    log_valid_file = None

    if opt.log:
        log_train_file = opt.log + '.train.log'
        log_valid_file = opt.log + '.valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

    valid_accus = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(model, training_data, optimizer, device, smoothing=opt.label_smoothing)
        print('  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, ' \
              'elapse: {elapse:3.3f} min'.format(
            ppl=math.exp(min(train_loss, 100)), accu=100 * train_accu,
            elapse=(time.time() - start) / 60))

        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, device)
        print('  - (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, ' \
              'elapse: {elapse:3.3f} min'.format(
            ppl=math.exp(min(valid_loss, 100)), accu=100 * valid_accu,
            elapse=(time.time() - start) / 60))

        valid_accus += [valid_accu]

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i}

        if opt.save_model:
            if opt.save_mode == 'all':
                model_name = opt.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100 * valid_accu)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = opt.save_model + '.chkpt'
                if valid_accu >= max(valid_accus):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100 * train_accu))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100 * valid_accu))


def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-data_train', default='data/name_train.pt')
    parser.add_argument('-data_val', default='data/name_val.pt')
    parser.add_argument('-data_all', default='data/data.pt')
    parser.add_argument('-data_set', default='data/data_set_error.pt')

    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=64)

    # parser.add_argument('-d_word_vec', type=int, default=512)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default='trained')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')
    parser.add_argument('-batch_x', default=64)
    parser.add_argument('-batch_y', default=32)
    parser.add_argument('-train_type', default='time')

    opt = parser.parse_args()
    opt.cuda = torch.cuda.is_available()
    opt.d_word_vec = opt.d_model

    # ========= Loading Dataset =========#
    # opt.max_token_seq_len = data['settings'].max_token_seq_len

    training_data, validation_data, train_time, val_time = get_data_loader(opt)

    opt.src_vocab_size = 728
    opt.tgt_vocab_size = 728
    if opt.train_type == 'time':
        opt.tgt_vocab_size = get_time_vac(opt)


# ========= Preparing Model =========#
    if opt.embs_share_weight:
        assert opt.src_vocab_size == opt.tgt_vocab_size, \
            'The src/tgt word2idx table are different but asked to share word embedding.'

    print(opt)

    transformer = Transformer(
        opt.src_vocab_size,
        opt.tgt_vocab_size,
        opt.batch_x + 2,
        tgt_emb_prj_weight_sharing=opt.proj_share_weight,
        emb_src_tgt_weight_sharing=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout).to(device)

    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, transformer.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        opt.d_model, opt.n_warmup_steps)
    if opt.train_type == 'name':
        train(transformer, training_data, validation_data, optimizer, device, opt)
    else:
        train(transformer, train_time, val_time, optimizer, device, opt)


def split_data_set(train_data_set, batch_x, batch_y, step_i=12):
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
            group_data.append(np.array(tmp))
        if len(tmp) % (batch_y + batch_x) == 0:
            group_data_y.append(np.array(tmp[batch_y * -1:]))
            tmp = []
            current_i = current_i + step_i
            step = current_i
    group_data.pop(-1)
    return torch.tensor(group_data).to(device), torch.tensor(group_data_y).to(device)


# 将时间处理成时间间隔
def time_split(train_data_time):
    c_time = train_data_time
    r_time = []
    for index, value in c_time.iteritems():
        if index == 0:
            r_time.append(0)
        else:
            get_sec = lambda x, y: (x - y).seconds if x > y else (y - x).seconds
            seconds = get_sec(c_time[index], c_time[index - 1])
            r_time.append(seconds)
    return r_time


def get_data_loader(opt):
    if os.path.exists(opt.data_set):
        data_loader = torch.load(opt.data_set)['train']
        data_loader_val = torch.load(opt.data_set)['val']
        train_loader_time = torch.load(opt.data_set)['time']
        val_loader_time = torch.load(opt.data_set)['val_time']
    else:
        data_train = torch.load(opt.data_all)['train_data']['dev_name']
        m_data = split_data_set(data_train, opt.batch_x, opt.batch_y)
        data_set_train = M_Test_data(m_data)
        data_loader = torch.utils.data.DataLoader(data_set_train, batch_size=opt.batch_size, shuffle=True,
                                                  pin_memory=True,
                                                  drop_last=True)
        data_val = torch.load(opt.data_all)['val_data']['dev_name']
        m_data_val = split_data_set(data_val, opt.batch_x, opt.batch_y)
        data_set_val = M_Test_data(m_data_val)
        data_loader_val = torch.utils.data.DataLoader(data_set_val, batch_size=opt.batch_size, shuffle=True,
                                                      pin_memory=True, drop_last=True)

        # 创建 网元-时间 训练集
        train_data = torch.load(opt.data_all)['train_data']['time']
        data_time = time_split(train_data)
        train_time_x, train_time_y = split_data_set(data_time, opt.batch_x, opt.batch_y)
        train_name_x, train_name_y = m_data
        m_data_time = train_name_x, train_time_y
        data_set_time = M_Test_data(m_data_time)
        train_loader_time = torch.utils.data.DataLoader(data_set_time, batch_size=opt.batch_size, shuffle=True,
                                                        pin_memory=True, drop_last=True)

        # 创建 网元-时间 测试集
        val_data_time = time_split(torch.load(opt.data_all)['val_data']['time'])
        val_time_x, val_time_y = split_data_set(val_data_time, opt.batch_x, opt.batch_y)
        val_name_x, val_name_y = m_data_val

        m_data_time = val_name_x, val_time_y
        data_set_time = M_Test_data(m_data_time)
        val_loader_time = torch.utils.data.DataLoader(data_set_time, batch_size=opt.batch_size, shuffle=True,
                                                      pin_memory=True, drop_last=True)

        data_loader_p = {
            'train': data_loader,
            'val': data_loader_val,
            'time': train_loader_time,
            'val_time': val_loader_time
        }
        torch.save(data_loader_p, opt.data_set)
    return data_loader, data_loader_val, train_loader_time, val_loader_time

def get_time_vac(opt):
    train_data = torch.load(opt.data_all)['train_data']['time']
    train_data = time_split(train_data)
    size = np.unique(np.array(train_data)).size
    return size


if __name__ == '__main__':
    main()

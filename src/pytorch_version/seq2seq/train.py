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
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim
from tensorboardX import SummaryWriter

import load_data as ld

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter('tfb')


def cal_performance(pred, gold, data_val_ofpa=None, smoothing=False):
    ''' Apply label smoothing if needed '''

    if data_val_ofpa is None:
        data_val_ofpa = []
    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1]
    post = gold.view(-1, 32)[:, -5:]
    gold = gold.contiguous().view(-1)

    post_pre = pred.view(-1, 32)[:, -5:]
    count_soba = 0
    for i in data_val_ofpa:
        count_soba += gold[gold == i].sum().item()



    post_correct = post_pre.eq(post).sum().item()
    # print(pred, gold)

    non_pad_mask = gold.ne(Constants.PAD)

    count_soba_pred = 0
    for i in data_val_ofpa:
        count_soba_pred += pred[gold == i].sum().item()


    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct, post_correct, count_soba, count_soba_pred


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
    step = 1
    n_post_correct = 0
    n_ofpa_all = 0
    n_ofpa_correct = 0

    for batch in tqdm(training_data, mininterval=2, desc='  - (Training)   ', leave=False):
        # prepare data
        src_seq, tgt_seq = map(lambda x: x.to(device).to(torch.int64), batch)
        #
        src_pos = torch.tensor(get_position(src_seq.shape)).to(device)
        tgt_pos = torch.tensor(get_position(tgt_seq.shape)).to(device)
        # gold = tgt_seq[:, 1:]

        # forward
        optimizer.zero_grad()
        pred = model(src_seq, src_pos, tgt_seq, tgt_pos)

        # backward
        loss, n_correct,post,ofpa_all,ofpa_correct = cal_performance(pred, tgt_seq, smoothing=smoothing)

        # print(loss)
        loss.backward()

        # update parameters
        optimizer.step_and_update_lr()

        # note keeping
        total_loss += loss.item()
        if (step % 50) == 0:
            writer.add_scalar('ACT-Train/loss/50step', loss.item(), step)
            writer.close()

        non_pad_mask = tgt_seq.ne(Constants.PAD)
        n_word = non_pad_mask.sum().item()
        n_word_total += n_word
        n_word_correct += n_correct
        n_post_correct += post
        n_ofpa_all += ofpa_all
        n_ofpa_correct += ofpa_correct

    loss_per_word = total_loss / n_word_total
    accuracy = n_word_correct / n_word_total
    if n_post_correct :
        ltpa = n_post_correct/(n_word_total/6)
    else:
        ltpa = 0
    if n_ofpa_correct:
        ofpa = n_ofpa_correct/n_ofpa_all
    else:
        ofpa = 0
    return loss_per_word, accuracy,ltpa,ofpa


def get_position(shape):
    pos = []
    pos_i = []
    for i in range(shape[1]):
        pos_i.append(i + 1)
    for i in range(shape[0]):
        pos.append(pos_i)
    return pos


def eval_epoch(model, validation_data, device,data_val_ofpa):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0
    n_post_correct =0
    n_ofpa_all =0
    n_ofpa_correct = 0
    with torch.no_grad():
        for batch in tqdm(
                validation_data, mininterval=2,
                desc='  - (Validating) ', leave=False):
            # prepare data
            src_seq, tgt_seq = map(lambda x: x.to(device).to(torch.int64), batch)
            # gold = tgt_seq[:, 1:]
            src_pos = torch.tensor(get_position(src_seq.shape)).to(device)
            tgt_pos = torch.tensor(get_position(tgt_seq.shape)).to(device)
            # forward
            pred = model(src_seq, src_pos, tgt_seq, tgt_pos)
            loss, n_correct, post,ofpa_all,ofpa_correct = cal_performance(pred, tgt_seq, data_val_ofpa,smoothing=False)

            # note keeping
            total_loss += loss.item()
            non_pad_mask = tgt_seq.ne(Constants.PAD)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct
            n_post_correct += post
            n_ofpa_all += ofpa_all
            n_ofpa_correct += ofpa_correct

    loss_per_word = total_loss / n_word_total
    accuracy = n_word_correct / n_word_total
    if n_post_correct :
        ltpa = n_post_correct/ (n_word_total/6)
    else:
        ltpa =0
    if n_ofpa_correct :
        ofpa = n_ofpa_correct/n_ofpa_all
    else:
        ofpa =0
    return loss_per_word, accuracy,ltpa,ofpa


def train(model, training_data, validation_data, optimizer, device, opt,data_val_ofpa):
    ''' Start training '''

    log_train_file = None
    log_valid_file = None

    if opt.log:
        log_train_file = opt.log + str(time.clock())+ '.train.log'
        log_valid_file = opt.log + str(time.clock())+ '.valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file,'a') as log_tf, open(log_valid_file, 'a') as log_vf:
            log_tf.write(str(opt)+'\n')
            log_vf.write(str(opt)+'\n')

        # with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
        #     log_tf.write('epoch,loss,ppl,accuracy\n')
        #     log_vf.write('epoch,loss,ppl,accuracy\n')

    valid_accus = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu,t_ltpa,t_ofpa = train_epoch(model, training_data, optimizer, device, smoothing=opt.label_smoothing)
        t_elapse =( time.time() - start) / 60
        print('  - (train epoch) loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, ' \
              'ltpa: {ltpa1:3.3f} %,ofpa: {ofpa1:3.3f} %,'
              'elapse: {elapse:3.3f} min'.format(
            loss=train_loss, accu=100 * train_accu, ltpa1=100 * t_ltpa, ofpa1=100 * t_ofpa,
            elapse=t_elapse))

        writer.add_scalar('ACT-Train/Loss', train_loss, epoch_i)
        writer.add_scalar('ACT-Train/Accuracy', 100 * train_accu, epoch_i)

        start = time.time()
        valid_loss, valid_accu,v_ltpa, v_ofpa = eval_epoch(model, validation_data, device,data_val_ofpa)
        print('  - (Validation epoch) loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, ' \
              'ltpa: {ltpa1:3.3f} %,ofpa: {ofpa1:3.3f} %,'
              'elapse: {elapse:3.3f} min'.format(
            loss=valid_loss, accu=100 * valid_accu,ltpa1=100*v_ltpa,ofpa1=100* v_ofpa,
            elapse=(time.time() - start) / 60))

        writer.add_scalar('ACT-valid/Accuracy', 100 * valid_accu, epoch_i)
        writer.close()

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
                log_tf.write('epoch: {epoch},loss: {loss: 8.5f}, {ppl: 8.5f}, SOBA:{SOBA:3.3f}, LTPA: {LTPA:3.3f}, OFPA: {OFPA:3.3f}, elapse:{t_elapse} end.\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)),
                    SOBA=100 * train_accu,LTPA = t_ltpa,OFPA=t_ofpa,t_elapse=t_elapse))
                log_vf.write('epoch: {epoch},loss: {loss: 8.5f}, {ppl: 8.5f}, SOBA:{SOBA:3.3f}, LTPA: {LTPA:3.3f}, OFPA: {OFPA:3.3f} end.\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), SOBA=100 * valid_accu,LTPA = v_ltpa,OFPA=v_ofpa))


def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    # parser.add_argument('-data_train', default='data/name_train.pt')
    # parser.add_argument('-data_val', default='data/name_val.pt')
    parser.add_argument('-data_all', default='data/csv/data_train_2_sort.torch')
    # parser.add_argument('-data_set', default='data/data_set/2018-06-01#2018-06-15.pt')
    # parser.add_argument('-torch_save_data', default='data/origin/2018-06-01#2018-06-15.pt')
    parser.add_argument('-save_model', default='module/2018-12-30.pt')
    parser.add_argument('-start_time', default='2018-06-01')
    parser.add_argument('-end_time', default='2018-11-01')

    parser.add_argument('-epoch', type=int, default=8)
    parser.add_argument('-batch_size', type=int, default=256)

    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=32)
    parser.add_argument('-d_v', type=int, default=32)

    parser.add_argument('-n_head', type=int, default=1)
    parser.add_argument('-n_layers', type=int, default=4)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-log', default='log/logs.log')

    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')
    parser.add_argument('-batch_x', default=32)
    parser.add_argument('-batch_y', default=30)
    parser.add_argument('-train_type', default='name')

    opt = parser.parse_args()
    opt.cuda = torch.cuda.is_available()
    opt.d_word_vec = opt.d_model

    # ========= Loading Dataset =========#
    # opt.max_token_seq_len = data['settings'].max_token_seq_len

    training_data, validation_data, voc_name,data_val_ofpa = ld.get_data_loader(opt, device)
    opt.src_vocab_size = voc_name
    opt.tgt_vocab_size = opt.src_vocab_size
    if opt.train_type == 'time':
        voc = ld.get_time_vac(opt)
        opt.tgt_vocab_size = voc if voc > 500 else 728

        # ========= Preparing Model =========#
    if opt.embs_share_weight:
        assert opt.src_vocab_size == opt.tgt_vocab_size, \
            'The src/tgt word2idx table are different but asked to share word embedding.'

    print(opt)

    transformer = Transformer(
        opt.src_vocab_size,
        opt.tgt_vocab_size,
        opt.batch_x,
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
    if opt.train_type == 'time':
        print("train time dim ")
        # train(transformer, train_time, val_time, optimizer, device, opt)
    else:
        train(transformer, training_data, validation_data, optimizer, device, opt,data_val_ofpa)



if __name__ == '__main__':
    main()

import pandas as pd
import torch

import Translator as tsl

csv_toPredict_noLabel = "data/csv/toPredict_noLabel.csv"
col_toPredict = ["id_sample", "id_road", "time"]
csv_toPredict_train_TTI = "data/csv/toPredict_train_TTI.csv"
col_train = ["id_road", "TTI", "speed", "time"]

opt = torch.load("data/dict/opt.torch")["opt"]


def get_csv(file_path, col_names):
    data = pd.read_csv(file_path, names=col_names, encoding='utf-8', skiprows=1)
    return data


def group_data(data):
    data = data.TTI.tolist()
    s = []
    i = 1
    input = []
    for tti in data:
        s.append(int(tti))
        if i % 6 == 0:
            input.append(s)
            s = []
        i += 1
    return input


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # tti = torch.load("data/csv/tti.torch")["tti"]
    # src = group_data(tti)
    # sl =[]
    # sl.append(src[0])
    # sl.append(src[1])
    # sl.append(src[2])
    # sl = torch.tensor(sl).to(device)
    # p = [1,2,3,4,5,6]
    # pl=[]
    # pl.append(p)
    # pl.append(p)
    # pl.append(p)
    # p = torch.tensor(p).to(device)
    ss = torch.load("sss.s")
    src_seq = ss["src_seq"]
    tgt_seq = ss["tgt_seq"]
    src_pos = ss["src_pos"]
    tgt_pos = ss["tgt_pos"]
    checkpoint = torch.load('module/d_int.pt_accu_99.928.chkpt')
    translator = tsl.Translator(checkpoint)
    batch_hyp, batch_scores = translator.translate_batch(src_seq, src_pos)
    print(batch_hyp)

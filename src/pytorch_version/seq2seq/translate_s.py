''' Translate input text with trained model. '''

import torch
import argparse
from tqdm import tqdm

import transformer.Constants as Constants
from transformer.Models import Transformer
import Translator_s


def load_model(opt, device):

    checkpoint = torch.load(opt.model, map_location=device)
    model_opt = checkpoint['settings']

    model = Transformer(
        model_opt.src_vocab_size,
        model_opt.trg_vocab_size,

        model_opt.src_pad_idx,
        model_opt.trg_pad_idx,

        trg_emb_prj_weight_sharing=model_opt.proj_share_weight,
        emb_src_trg_weight_sharing=model_opt.embs_share_weight,
        d_k=model_opt.d_k,
        d_v=model_opt.d_v,
        d_model=model_opt.d_model,
        d_word_vec=model_opt.d_word_vec,
        d_inner=model_opt.d_inner_hid,
        n_layers=model_opt.n_layers,
        n_head=model_opt.n_head,
        dropout=model_opt.dropout).to(device)

    model.load_state_dict(checkpoint['model'])
    print('[Info] Trained model state loaded.')
    return model 


def main():
    '''Main Function'''

    parser = argparse.ArgumentParser(description='translate_s.py')

    parser.add_argument('-model', 'module/d_int.pt.chkpt')

    parser.add_argument('-output', default='pred.txt',
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    parser.add_argument('-beam_size', type=int, default=5)
    parser.add_argument('-max_seq_len', type=int, default=100)

    opt = parser.parse_args()

    data = pickle.load(open(opt.data_pkl, 'rb'))
    SRC, TRG = data['vocab']['src'], data['vocab']['trg']

    test_loader = Dataset(examples=data['test'], fields={'src': SRC, 'trg': TRG})
    
    device = torch.device('cuda')
    translator = Translator_s.to(device)

    with open(opt.output, 'w') as f:
        for example in tqdm(test_loader, mininterval=2, desc='  - (Test)', leave=False):
            #print(' '.join(example.src))
            src_seq = [SRC.vocab.stoi.get(word, unk_idx) for word in example.src]
            pred_seq = translator.translate_sentence(torch.LongTensor([src_seq]).to(device))
            pred_line = ' '.join(TRG.vocab.itos[idx] for idx in pred_seq)
            pred_line = pred_line.replace(Constants.BOS_WORD, '').replace(Constants.EOS_WORD, '')
            #print(pred_line)
            f.write(pred_line.strip() + '\n')

    print('[Info] Finished.')

if __name__ == "__main__":
    '''
    Usage: python translate_s.py -model trained.chkpt -data multi30k.pt -no_cuda
    '''
    main()

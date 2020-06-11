''' This module will handle the text generation with beam search. '''

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.Beam import Beam


class Translator(object):
    ''' Load with trained model and handle the beam search '''

    def __init__(self, model):
        self.device = torch.device('cuda')

        model.word_prob_prj = nn.LogSoftmax(dim=1)

        model = model.to(self.device)

        self.model = model
        self.model.eval()

    def translate_batch(self, src_seq, src_pos):
        ''' Translation work in one batch '''


        with torch.no_grad():
            # -- Encode
            src_seq, src_pos = src_seq[0].to(self.device), src_pos[0].to(self.device)
            src_enc, *_ = self.model.encoder(src_seq, src_pos)



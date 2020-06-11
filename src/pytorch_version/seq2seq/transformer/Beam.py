""" Manage beam search info structure.

    Heavily borrowed from OpenNMT-py.
    For code in OpenNMT-py, please check the following link:
    https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/Beam.py
"""

import torch

import transformer.Constants as Constants


class Beam():
    ''' Beam search '''

    def __init__(self, size, device=False):

        self.size = size
        self._done = False

        # The score for each translation on the beam.
        self.scores = torch.zeros((size,), dtype=torch.float, device=device)
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [torch.full((size,), Constants.PAD, dtype=torch.long, device=device)]
        # self.next_ys[0][0] = Constants.BOS

    def get_current_state(self,f_s):
        "Get the outputs for the current timestep."
        return self.get_tentative_hypothesis(f_s)

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    @property
    def done(self):
        return self._done

    def advance(self, word_prob):
        "Update beam status and check if finished or not."
        num_words = word_prob.size(1)

        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_lk = word_prob + self.scores.unsqueeze(1).expand_as(word_prob)
        else:
            beam_lk = word_prob[0]

        flat_beam_lk = beam_lk.view(-1)
        # flat_beam_lk = flat_beam_lk.masked_select(flat_beam_lk.ne(0))
        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True)
        zero_size = best_scores.masked_select(best_scores.eq(0)).size()
        if zero_size[0]>0:
            best_scores, best_scores_id = flat_beam_lk.topk(self.size+1, 0, True, True)
            best_scores = best_scores[best_scores!=0]
            best_scores_id = torch.cat([best_scores_id[0:0], best_scores_id[1:]])
        self.all_scores.append(self.scores)
        self.scores = best_scores

        # bestScoresId is flattened as a (beam x word) array,
        # so we need to calculate which word and beam each score came from
        prev_k = best_scores_id / num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append(best_scores_id - prev_k * num_words)

        # End condition is when top-of-beam is EOS.
        # if self.next_ys[-1][0].item() == Constants.EOS:
        #     self._done = True
        #     self.all_scores.append(self.scores)
        #
        # return self._done

    def sort_scores(self):
        "Sort the scores."
        return torch.sort(self.scores, 0, True)

    def get_the_best_score_and_idx(self):
        "Get the score of the best in the beam."
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_tentative_hypothesis(self,f_s):
        "Get the decoded sequence for the current timestep."

        if len(self.next_ys) == 1:
            self.next_ys[0][0] = f_s
            dec_seq = self.next_ys[0].unsqueeze(1)
        else:
            _, keys = self.sort_scores()
            hyps = [self.get_hypothesis(k) for k in keys]
            hyps = [[Constants.BOS] + h for h in hyps]
            dec_seq = torch.LongTensor(hyps)

        return dec_seq

    def get_hypothesis(self, k):
        """ Walk back to construct the full hypothesis. """
        hyp = []
        for j in range(len(self.prev_ks) - 1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])
            k = self.prev_ks[j][k]

        return list(map(lambda x: x.item(), hyp[::-1]))

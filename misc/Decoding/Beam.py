import torch
from config import Constants

class Beam():
    ''' Beam search '''

    def __init__(self, size, max_len, device=False, specific_nums_of_sents=0, begin_of_sentence_id=Constants.BOS):

        self.size = size
        self.specific_nums_of_sents = max(self.size, specific_nums_of_sents)
        self._done = False
        self.max_len=max_len

        # The score for each translation on the beam.
        self.scores = torch.zeros((size,), dtype=torch.float, device=device)
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [torch.full((size,), begin_of_sentence_id, dtype=torch.long, device=device)]

        self.finished = []

    def get_current_state(self):
        "Get the outputs till the current timestep."
        return self.get_tentative_hypothesis()

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    @property
    def done(self):
        return self._done

    def append_one_item(self, content):
        self.finished.append(content)
        if len(self.finished) >= self.specific_nums_of_sents:
            return True
        else:
            return False

    def advance(self, word_prob):
        "Update beam status and check if finished or not."
        num_words = word_prob.size(1)

        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_lk = word_prob + self.scores.unsqueeze(1).expand_as(word_prob)
            for i in range(self.next_ys[-1].size(0)):
                if self.next_ys[-1][i] == Constants.EOS:
                    beam_lk[i] = -1e20
        else:
            beam_lk = word_prob[0]

        flat_beam_lk = beam_lk.view(-1)

        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True)

        self.all_scores.append(self.scores)
        self.scores = best_scores

        # bestScoresId is flattened as a (beam x word) array,
        # so we need to calculate which word and beam each score came from
        prev_k = best_scores_id // num_words
        self.prev_ks.append(prev_k)
        #print('PREV_KS:', self.prev_ks)
        self.next_ys.append(best_scores_id - prev_k * num_words)

        for i in range(self.next_ys[-1].size(0)):
            if self.next_ys[-1][i] == Constants.EOS:
                self._done = self.append_one_item([self.scores[i].item(), len(self.prev_ks), i]) # (score, timestep, beam_id)
            if self._done:
                return self._done

        # End condition is when there has been collected self.specific_nums_of_sents predictions.
        if len(self.next_ys) == self.max_len:
            self._done = True
            self.all_scores.append(self.scores)
            if not len(self.finished):
                for i in range(self.next_ys[-1].size(0)):
                    self.append_one_item([self.scores[i].item(), len(self.prev_ks), i])
        return self._done

    def sort_scores(self):
        "Sort the scores."
        return torch.sort(self.scores, 0, True)

    def sort_finished(self, alpha=1.0):
        for item in self.finished:
            # a regularization on sentence length
            # if alpha > 1, it prefers longer captions
            # if alpha < 1, it prefers shorter captions
            item[0] /= item[1]**alpha 

        self.finished.sort(key=lambda a: -a[0])
        scores = [sc for sc, _, _ in self.finished]
        tk = [(t, k) for _, t, k in self.finished]
        return scores, tk

    def get_hypothesis_from_tk(self, timestep, k):
        """ Walk back to construct the full hypothesis. """
        return self.get_hypothesis(k, add_bos=False, length=timestep)

    def get_the_best_score_and_idx(self):
        "Get the score of the best in the beam."
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_tentative_hypothesis(self):
        "Get the decoded sequence for the current timestep."
        _, keys = self.sort_scores()
        hyps = [self.get_hypothesis(k, add_bos=True) for k in keys]
        dec_seq = torch.LongTensor(hyps)
        return dec_seq

    def get_hypothesis(self, k, add_bos=False, length=None):
        """ Walk back to construct the full hypothesis. """
        if length is None:
            length = len(self.prev_ks)

        hyp = []
        for j in range(length - 1, -1, -1):
            hyp.append(self.next_ys[j+1][k])
            k = self.prev_ks[j][k]
        
        if add_bos:
            hyp.append(self.next_ys[0][k])

        return list(map(lambda x: x.item(), hyp[::-1]))

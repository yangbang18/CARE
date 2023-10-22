import math
import torch
import torch.nn as nn
from config import Constants
from .base import CritBase
from ..logger import AverageMeter


class LanguageGeneration(CritBase):
    def __init__(self, opt):
        visual_word_generation = opt.get('visual_word_generation', False)

        if visual_word_generation:
            weights = opt.get('nv_weights', [0.8, 1.0])
            self.num_word_acc = 2
        else:
            weights = 1.0
            self.num_word_acc = 1
        
        super().__init__(keys=['logits', 'labels', 'probs'], weights=weights)
        self.label_smoothing = opt["label_smoothing"]
        self.loss_fn = nn.NLLLoss(reduction='none')
        self.ignore_index = Constants.PAD
        self.visual_word_generation = visual_word_generation
        self.opt = opt

    def _step(self, 
            index_indicator: int, 
            logits: torch.Tensor, 
            labels: torch.Tensor, 
            probs: torch.Tensor = None,
            *others,
        ):
        """
            args:
                logits: [batch_size, seq_len, vocab_size]
                labels: [batch_size, seq_len]
        """
        if probs is not None:
            logits = probs

        assert not len(others)
        if (self.opt.get('use_attr', False)) and 'prefix' in self.opt.get('use_attr_type', ''):
            assert logits.size(1) == labels.size(1) + self.opt['use_attr_topk']
            logits = logits[:, self.opt['use_attr_topk']:, :]
        elif (self.opt.get('use_attr', False)) and 'pp' in self.opt.get('use_attr_type', ''):
            assert logits.size(1) == labels.size(1) + 1
            logits = logits[:, 1:, :]
        elif logits.size(1) == labels.size(1) + 1:
            logits = logits[:, :-1, :]
        else:
            assert logits.size(1) == labels.size(1)

        if probs is not None:
            tgt_word_logprobs = (logits + 1e-6).log()
        else:
            tgt_word_logprobs = torch.log_softmax(logits, dim=-1)

        # calculate the top-1 accuracy of the generated words
        self.calculate_word_acc(index_indicator, tgt_word_logprobs, labels)
        # calculate the perplexity of the generated words
        self.calculate_perplexity(index_indicator, tgt_word_logprobs, labels)

        tgt_word_logprobs = tgt_word_logprobs.contiguous().view(-1, tgt_word_logprobs.size(2))
        labels = labels.contiguous().view(-1)
        loss = (1 - self.label_smoothing) * self.loss_fn(tgt_word_logprobs, labels) + \
               self.label_smoothing * - tgt_word_logprobs.mean(dim=-1)

        if self.ignore_index is not None:
            mask = labels.ne(self.ignore_index).float()
            return torch.sum(loss * mask)
        else:
            return torch.sum(loss)
    
    def calculate_word_acc(self, index_indicator, preds, gts):
        ind = gts.ne(Constants.PAD)
        if index_indicator == 0 and self.visual_word_generation:
            ind = ind & gts.ne(Constants.MASK)
        
        predict_res = preds.max(-1)[1][ind]
        target_res = gts[ind]

        self.word_acc_recorder[index_indicator].update(
                    (predict_res == target_res).sum().item(),
                    predict_res.size(0), 
                    multiply=False
            )

    def calculate_perplexity(self, index_indicator, preds, gts):
        # for the methods with visual word generation
        # we only compute the perplexity of the caption genration process
        if index_indicator == 0 and self.visual_word_generation:
            return None

        assert len(preds.shape) == 3
        assert preds.shape[:-1] == gts.shape

        log_probs = preds.gather(2, gts.unsqueeze(2)).squeeze(2)
        mask = gts.ne(Constants.PAD)
        num_words = float(torch.sum(mask))

        per_word_cross_entropy = -torch.sum(log_probs * mask) / num_words
        self.perplexity_recorder.update(per_word_cross_entropy.item(), num_words)

    def get_fieldsnames(self):
        return ['Word Acc%d' % i for i in range(self.num_word_acc)] + ['Perplexity']

    def get_info(self):
        info = [meter.avg for meter in self.word_acc_recorder]
        info += [math.exp(self.perplexity_recorder.avg)]
        return self.get_fieldsnames(), info

    def reset_recorder(self):
        self.word_acc_recorder = [AverageMeter() for _ in range(self.num_word_acc)]
        self.perplexity_recorder = AverageMeter()

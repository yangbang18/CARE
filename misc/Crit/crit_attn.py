import torch
import torch.nn as nn
from .base import CritBase
from config import Constants


class AttnSparseLoss(CritBase):
    def __init__(self, opt):
        super().__init__(keys=['attr_attention_probs', 'labels', 'attribute_mask'], batch_mean=True)
        self.crit = nn.L1Loss(reduction='none')
        self.threshold = opt.get('use_attr_attn_loss_threshold', 1)
        self.use_attribute_mask = opt.get('use_attr_attn_loss_mask', False)
    
    def _step(self, index_indicator, attr_attention_probs, labels, attribute_mask, *others):
        assert attr_attention_probs.dim() == 4, attr_attention_probs.shape
        assert attr_attention_probs.size(2) == labels.size(1)

        attr_attention_probs = attr_attention_probs.sum(-1) # (bsz, n_heads, seq_len)
        attr_attention_probs = attr_attention_probs.mean(1) # (bsz, seq_len)

        assert attr_attention_probs.shape == labels.shape

        mask = labels.eq(Constants.PAD)
        attr_attention_probs[mask] = self.threshold

        target = labels.new(*labels.shape).data.fill_(self.threshold)

        if self.use_attribute_mask:
            assert attribute_mask.shape == labels.shape
            target[attribute_mask.eq(0)] = 0
        
        attr_attention_probs[attr_attention_probs < target] = self.threshold

        loss = self.crit(attr_attention_probs, target) # (bsz, seq_len)

        loss = torch.sum(loss * (~mask), dim=1) / (~mask).sum(1)

        return torch.sum(loss)


class GateLoss(CritBase):
    def __init__(self, opt):
        wise = opt.get('attentive_loss_wise', False)
        super().__init__(keys=['labels', 'non_stop_words_mask', 'gate_probs'], batch_mean=not wise)
        self.wise = wise
    
    def _step(self, index_indicator, labels, non_stop_words_mask, gate_probs, *others):
        assert labels.shape == non_stop_words_mask.shape

        device = gate_probs[0].device
        valid_mask = labels.ne(Constants.PAD).contiguous().view(-1).to(device)
        target = non_stop_words_mask.contiguous().view(-1).float().to(device)

        loss = 0
        for probs in gate_probs:
            assert probs.shape[:2] == labels.shape
            probs = probs.mean(2).contiguous().view(-1) # (bsz * seq_len, )
            this_loss = -(target * torch.log(probs) + (1.0 - target) * torch.log(1.0 - probs))
            loss = loss + this_loss
        
        loss = (loss * valid_mask).sum()

        if self.wise:
            loss = loss / valid_mask.sum().float()

        return loss

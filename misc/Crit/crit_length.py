import torch
import torch.nn as nn
from .base import CritBase


class KLDivLoss(CritBase):
    def __init__(self, opt):
        super().__init__(keys=['preds_length', 'length_target'], batch_mean=True)
        self.crit = nn.KLDivLoss(reduction='none')
    
    def _step(self, index_indicator, preds_length, length_target, *others):
        loss = self.crit(preds_length, length_target.to(preds_length.device))
        return torch.sum(loss)

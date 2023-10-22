import torch
import torch.nn as nn
from typing import Dict, List, Union    


class Predictor(nn.Module):
    def __init__(self, nets: List[nn.Module]):
        super().__init__()
        self.nets = nn.ModuleList(nets)

    def forward(self, encoder_hidden_states: Union[torch.Tensor, List[torch.Tensor]], **kwargs) -> Dict[str, torch.Tensor]:
        results = {}
        for net in self.nets:
            results.update(net(encoder_hidden_states, **kwargs, **results))
        return results

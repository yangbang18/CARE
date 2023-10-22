import torch
from torch import nn

class CompositionalLinear(nn.Module):
    def __init__(self, dim_hidden, dim_factor, dim_semantic, dim_input, bias=True):
        super().__init__()
        self.linear_a = nn.Linear(dim_factor, dim_hidden, bias=False)
        self.linear_b = nn.Linear(dim_semantic, dim_factor, bias=False)
        self.linear_c = nn.Linear(dim_input, dim_factor, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(dim_hidden))
    
    def forward(self, input, semantic_input):
        out_b = self.linear_b(semantic_input).unsqueeze(1) # (bsz, dim_semantic) -> (bsz, 1, dim_factor)
        out_c = self.linear_c(input) # (bsz, L, dim_input) -> (bsz, L, dim_factor)
        out = self.linear_a(out_b * out_c) # (bsz, L, dim_factor) -> (bsz, L, dim_hidden)
        if hasattr(self, 'bias'):
            out = out + self.bias
        return out

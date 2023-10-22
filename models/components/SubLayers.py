''' Define sublayers in the encoder/decoder layer of Transformer'''

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from .Attention import ScaledDotProductAttention, CompositionalSDPA
from .activations import get_activation
from .basic import CompositionalLinear


class MultiHeadAttention(nn.Module):
    def __init__(self, 
            dim_hidden: int,             
            hidden_dropout_prob: float = 0.5, 
            has_ln: bool = True,
            pre_ln: bool = False,
            layer_norm_eps: float = 1e-12,
            skip_connection: bool = True,
            attention_class = ScaledDotProductAttention,
            **kwargs,
        ):
        super(MultiHeadAttention, self).__init__()
        self.SDPA = attention_class(dim_hidden=dim_hidden, **kwargs)
        
        self.dim_hidden = dim_hidden
        self.define_dense()    
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(dim_hidden, eps=layer_norm_eps) if has_ln else None
        
        self.pre_ln = pre_ln
        self.skip_connection = skip_connection
    
    def define_dense(self):
        self.dense = nn.Linear(self.dim_hidden, self.dim_hidden)

    def forward_dense(self, hidden_states, **kwargs):
        context = self.dense(hidden_states)
        return context
        
    def forward(self, 
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            input_tensor: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None, 
            head_mask: Optional[torch.Tensor] = None,
            q: Optional[torch.Tensor] = None,
            k: Optional[torch.Tensor] = None,
            v: Optional[torch.Tensor] = None,
            early_return: bool = False,
            **kwargs
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        if input_tensor is None:
            input_tensor = hidden_states.clone()

        if self.pre_ln and self.LayerNorm is not None:
            hidden_states = self.LayerNorm(hidden_states)
        
        if q is not None:
            assert k is not None
            assert v is not None
        else:
            if encoder_hidden_states is None:
                q = k = v = hidden_states
            else:
                q = hidden_states
                k = v = encoder_hidden_states

        hidden_states, attention_probs = self.SDPA(q, k, v, attention_mask, head_mask, **kwargs)
        context = self.forward_dense(hidden_states, **kwargs)
        context = self.dropout(context)

        if early_return:
            return context, attention_probs

        if self.skip_connection:
            hidden_states = context + input_tensor
        
        if not self.pre_ln and self.LayerNorm is not None:
            hidden_states = self.LayerNorm(hidden_states)

        return hidden_states, attention_probs, context


class GatedMultiHeadAttention(MultiHeadAttention):
    def __init__(self, dim_hidden, scalar_gate=False, **kwargs):
        super().__init__(dim_hidden, **kwargs)
        self.gate = nn.Sequential(
            nn.Linear(dim_hidden * 2, 1 if scalar_gate else dim_hidden),
            nn.Sigmoid()
        )
    
    def forward(self, 
            hidden_states: torch.Tensor,
            **kwargs
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        context, attention_probs = super().forward(hidden_states, early_return=True, **kwargs)
        gate_probs = self.gate(torch.cat([hidden_states, context], dim=-1))

        hidden_states = hidden_states + gate_probs * context
            
        if not self.pre_ln:
            hidden_states = self.LayerNorm(hidden_states)
        
        return hidden_states, (attention_probs, gate_probs), context
        

class PositionwiseFeedForward(nn.Module):
    def __init__(self, 
            dim_hidden: int, 
            dim_intermediate: int, 
            hidden_act: str = 'relu',
            hidden_dropout_prob: float = 0.5, 
            layer_norm_eps: float = 1e-12,
            pre_ln: bool = False,
            **kwargs,
        ):
        super(PositionwiseFeedForward, self).__init__()
        self.dim_hidden = dim_hidden
        self.dim_intermediate = dim_intermediate
        self.define_dense()
        self.act = get_activation(hidden_act)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(dim_hidden, eps=layer_norm_eps)
        self.pre_ln = pre_ln
    
    def define_dense(self):
        self.dense1 = nn.Linear(self.dim_hidden, self.dim_intermediate)
        self.dense2 = nn.Linear(self.dim_intermediate, self.dim_hidden)
    
    def forward_dense1(self, hidden_states, **kwargs):
        return self.dense1(hidden_states)

    def forward_dense2(self, hidden_states, **kwargs):
        return self.dense2(hidden_states)

    def forward(self, hidden_states, **kwargs):
        input_tensor = hidden_states.clone()

        if self.pre_ln:
            hidden_states = self.LayerNorm(hidden_states)

        hidden_states = self.forward_dense1(hidden_states, **kwargs)
        hidden_states = self.act(hidden_states)
        hidden_states = self.forward_dense2(hidden_states, **kwargs)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor
        if not self.pre_ln:
            hidden_states = self.LayerNorm(hidden_states)
        
        return hidden_states


class CompositionalMHA(MultiHeadAttention):
    def __init__(self, **kwargs):
        self.dim_factor = kwargs['dim_hidden'] // kwargs.get('dim_factor_scale', 2)
        self.dim_semantic = kwargs['dim_semantic']
        super().__init__(**kwargs, attention_class=CompositionalSDPA)
    
    def define_dense(self):
        self.dense = CompositionalLinear(self.dim_hidden, self.dim_factor, self.dim_semantic, self.dim_hidden)
    
    def forward_dense(self, hidden_states, preds_attr, **kwargs):
        return self.dense(hidden_states, preds_attr.detach())


class CompositionalFFN(PositionwiseFeedForward):
    def __init__(self, **kwargs):
        self.dim_factor = kwargs['dim_hidden'] // kwargs.get('dim_factor_scale', 2)
        self.dim_semantic = kwargs['dim_semantic']
        super().__init__(**kwargs)
    
    def define_dense(self):
        self.dense1 = CompositionalLinear(self.dim_intermediate, self.dim_factor, self.dim_semantic, self.dim_hidden)
        self.dense2 = CompositionalLinear(self.dim_hidden, self.dim_factor, self.dim_semantic, self.dim_intermediate)
    
    def forward_dense1(self, hidden_states, preds_attr, **kwargs):
        return self.dense1(hidden_states, preds_attr.detach())
    
    def forward_dense2(self, hidden_states, preds_attr, **kwargs):
        return self.dense2(hidden_states, preds_attr.detach())

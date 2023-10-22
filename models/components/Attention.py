''' Define the common attention mechanisms '''

import torch
import torch.nn as nn
import math
from typing import Dict, Tuple, Union, List, Optional
from misc.utils import enlarge
from .basic import CompositionalLinear


class ScaledDotProductAttention(nn.Module):
    def __init__(self, 
            dim_hidden: int, 
            dim_key: Optional[int] = None,
            dim_value: Optional[int] = None,
            num_attention_heads: int = 1, 
            attention_probs_dropout_prob: float = 0.0,
            exclude_bias: bool = False,
            have_relative_position_bias: bool = False,
            max_relative_position: Optional[int] = None,
            attend_to_video: bool = False,
            add_hybrid_attention_bias: bool = False,
            hybrid_length: int = 0,
            **kwargs,
        ):
        super(ScaledDotProductAttention, self).__init__()

        self.dim_hidden = dim_hidden
        self.dim_key = dim_hidden if dim_key is None else dim_key
        self.dim_value = dim_hidden if dim_value is None else dim_value
        self.bias = not exclude_bias

        self.define_query_key_value()

        assert dim_hidden % num_attention_heads == 0
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = dim_hidden // num_attention_heads
        self.all_head_size = dim_hidden

        if have_relative_position_bias:
            assert max_relative_position is not None
            from .Embeddings import RelativePositionalEmbedding
            self.rpe = RelativePositionalEmbedding(max_relative_position, num_attention_heads, attend_to_video)
        
        self.dropout = nn.Dropout(attention_probs_dropout_prob)

        if add_hybrid_attention_bias:
            assert attend_to_video
            assert hybrid_length
            # self.hybrid_bias = nn.Parameter(torch.zeros(hybrid_length))
            self.hybrid_bias = nn.Parameter(torch.zeros(num_attention_heads, hybrid_length))
    
    def define_query_key_value(self):
        self.query = nn.Linear(self.dim_hidden, self.dim_hidden, bias=self.bias)
        self.key = nn.Linear(self.dim_key, self.dim_hidden, bias=self.bias)
        self.value = nn.Linear(self.dim_value, self.dim_hidden, bias=self.bias)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape) # [bsz, seq_len, n_head, head_size]
        return x.permute(0, 2, 1, 3) # [bsz, n_head, seq_len, head_size]

    def prepare_query_key_value(self, q, k, v, **kwargs):
        query = self.transpose_for_scores(self.query(q)) # [bsz, n_head, lq, head_size]
        key = self.transpose_for_scores(self.key(k))     # [bsz, n_head, lk, head_size]
        value = self.transpose_for_scores(self.value(v)) # [bsz, n_head, lv, head_size]
        return query, key, value

    def forward(self, 
            q: torch.Tensor, 
            k: torch.Tensor, 
            v: torch.Tensor, 
            attention_mask: Optional[torch.Tensor] = None, 
            head_mask: Optional[torch.Tensor] = None,
            return_attention_scores: bool = False,
            use_sigmoid_rather_than_softmax: bool = False,
            sigmoid_normalize: bool = False,
            **kwargs,
        ) -> Tuple[torch.Tensor, torch.Tensor]:

        query, key, value = self.prepare_query_key_value(q, k, v, **kwargs)
        
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) # [bsz, n_head, lq, lk]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if hasattr(self, 'rpe'):
            if kwargs['attend_to_video']:
                length_k = kwargs['n_frames']
            else:
                length_k = key.shape[-2]

            position_bias = self.rpe(
                length_q=query.shape[-2], 
                length_k=length_k, 
                device=query.device, 
                bidirectional=(kwargs['decoding_type'] == 'NARFormer')
            ) # [1, n_head, lq, lk]

            if kwargs['attend_to_video']:
                position_bias = position_bias.repeat(1, 1, 1, key.shape[-2] // kwargs['n_frames'])

            attention_scores = attention_scores + position_bias

        if attention_mask is not None:
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1) # [bsz, 1, lq, lk]
            attention_scores = attention_scores.masked_fill(attention_mask, -1e9)
        
        if hasattr(self, 'hybrid_bias'):
            # attention_scores = attention_scores + self.hybrid_bias[None, None, None, :]
            attention_scores = attention_scores + self.hybrid_bias[None, :, None, :]

        if use_sigmoid_rather_than_softmax:
            attention_probs = torch.sigmoid(attention_scores)
            if sigmoid_normalize:
                attention_probs = attention_probs / attention_probs.sum(dim=-1, keepdim=True) # normalize
        else:
            attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs) # [bsz, n_head, lq, lk]

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context = torch.matmul(attention_probs, value) # [bsz, n_head, lq, head_size]
        context = context.permute(0, 2, 1, 3).contiguous() # [bsz, lq, n_head, head_size]

        new_context_shape = context.size()[:-2] + (self.all_head_size,)
        context = context.view(*new_context_shape) # [bsz, lq, dim_hidden]

        return context, attention_probs if not return_attention_scores else attention_scores


class AdditiveAttention(nn.Module):
    def __init__(self, 
            dim_hidden: int, 
            dim_feats: Union[List[int], int], 
            dim_mid: int, 
            feats_share_weights: bool = False,
            add_hybrid_attention_bias: bool = False,
            hybrid_length: int = 0,
        ):
        super().__init__()
        self.linear1_h = nn.Linear(dim_hidden, dim_mid, bias=True)

        if not isinstance(dim_feats, list):
            dim_feats = [dim_feats]

        if feats_share_weights:    
            # check valid
            if feats_share_weights:
                for dim in dim_feats[1:]:
                    assert dim_feats[0] == dim
            dim_feats = [dim_feats[0]]

        self.linear1_f = nn.ModuleList([nn.Linear(dim, dim_mid) for dim in dim_feats])
        self.activation = nn.Tanh()
        self.linear2 = nn.Linear(dim_mid, 1, bias=False)

        if add_hybrid_attention_bias:
            self.hybrid_bias = nn.Parameter(torch.zeros(1, hybrid_length))

    def forward(self, 
            hidden_states: torch.Tensor, 
            feats: Union[List[torch.Tensor], torch.Tensor],
            return_raw: bool = False,
            return_attention_scores: bool = False,
            **kwargs
        ) -> Union[Dict[str, torch.Tensor], Tuple[List[torch.Tensor], List[torch.Tensor]]]:

        if len(hidden_states.shape) == 1:
            hidden_states = hidden_states.unsqueeze(0)
        if len(hidden_states.shape) == 3 and hidden_states.shape[0] == 1:
            hidden_states = hidden_states.squeeze(0)
        if not isinstance(feats, list):
            feats = [feats]

        assert len(self.linear1_f) == 1 or len(self.linear1_f) == len(feats)

        probs = []
        context = []
        bsz, seq_len, *_ = feats[-1].shape
        
        emb_h = enlarge(self.linear1_h(hidden_states), seq_len) # [bsz*seq_len, dim_mid]
        for i, inputs in enumerate(feats):
            layer = self.linear1_f[min(i, len(self.linear1_f)-1)]

            emb_f = layer(inputs).contiguous().view(bsz*seq_len, -1) # [bsz*seq_len, dim_mid]
            emb = self.activation(emb_h + emb_f)
            
            logits = self.linear2(emb).view(bsz, seq_len) # [bsz, seq_len]

            if hasattr(self, 'hybrid_bias'):
                logits = logits + self.hybrid_bias
            
            this_probs = torch.softmax(logits, dim=1)
            this_context = torch.bmm(this_probs.unsqueeze(1), inputs).squeeze(1) # [bsz, dim_feat]

            probs.append(this_probs if not return_attention_scores else logits)
            context.append(this_context)
        
        if return_raw:
            return context, probs

        # return context [bsz, sum(dim_feats)] & attention_probs [bsz, num_feats, seq_len]
        return torch.cat(context, dim=1), torch.stack(probs, dim=1)


class MultiLevelAttention(nn.Module):
    def __init__(self, 
            dim_hidden: int, 
            dim_feats: List[int], 
            dim_mid: int, 
            feats_share_weights: bool = False
        ):
        super().__init__()
        assert isinstance(dim_feats, list)
        assert len(dim_feats) > 1
        for dim in dim_feats[1:]:
            assert dim_feats[0] == dim

        self.temporal_aware_attention = AdditiveAttention(dim_hidden, dim_feats, dim_mid, feats_share_weights)
        self.modality_aware_attention = AdditiveAttention(dim_hidden, dim_feats[0], dim_mid)
    
    def forward(self, 
            hidden_states: torch.Tensor, 
            feats: List[torch.Tensor],
            **kwargs
        ) -> Dict[str, torch.Tensor]:

        context, probs = self.temporal_aware_attention(hidden_states, feats, return_raw=True, **kwargs)
        context = torch.stack(context, dim=1) # [bsz, num_feats, dim]

        context_2nd, probs_2nd = self.modality_aware_attention(hidden_states, context, return_raw=True, **kwargs)

        # return context [bsz, dim_feats[0]] & attention_probs [bsz, num_feats + 1, seq_len]
        return context_2nd, torch.stack(probs + probs_2nd, dim=1)


class CompositionalSDPA(ScaledDotProductAttention):
    def __init__(self, **kwargs):
        self.dim_factor = kwargs['dim_hidden'] // kwargs.get('dim_factor_scale', 2)
        self.dim_semantic = kwargs['dim_semantic']
        super().__init__(**kwargs)
    
    def define_query_key_value(self):
        self.query = CompositionalLinear(self.dim_hidden, self.dim_factor, self.dim_semantic, self.dim_hidden)
        self.key = CompositionalLinear(self.dim_hidden, self.dim_factor, self.dim_semantic, self.dim_key)
        self.value = CompositionalLinear(self.dim_hidden, self.dim_factor, self.dim_semantic, self.dim_value)

    def prepare_query_key_value(self, q, k, v, preds_attr, **kwargs):
        query = self.transpose_for_scores(self.query(q, preds_attr.detach())) # [bsz, n_head, lq, head_size]
        key = self.transpose_for_scores(self.key(k, preds_attr.detach()))     # [bsz, n_head, lk, head_size]
        value = self.transpose_for_scores(self.value(v, preds_attr.detach())) # [bsz, n_head, lv, head_size]
        return query, key, value

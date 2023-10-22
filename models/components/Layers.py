''' Define the layers in Transformer'''

import copy
import torch
import torch.nn as nn
from models.components.SubLayers import (
    MultiHeadAttention, 
    GatedMultiHeadAttention,
    CompositionalMHA,
    PositionwiseFeedForward,
    CompositionalFFN,
)
from typing import Dict, Any, Optional, Tuple


class EncoderLayer(nn.Module):
    def __init__(self, opt: Dict[str, Any]):
        super(EncoderLayer, self).__init__()
        self.intra_attention = MultiHeadAttention(
            dim_hidden=opt['dim_hidden'],
            num_attention_heads=opt['num_attention_heads'],
            attention_probs_dropout_prob=opt['attention_probs_dropout_prob'],
            hidden_dropout_prob=opt['hidden_dropout_prob'],
            layer_norm_eps=opt['layer_norm_eps'],
            exclude_bias=opt.get('mha_exclude_bias', False),
            pre_ln=opt.get('transformer_pre_ln', False),
        )
        self.ffn = PositionwiseFeedForward(
            dim_hidden=opt['dim_hidden'],
            dim_intermediate=opt['intermediate_size'],
            hidden_act=opt['hidden_act'],
            hidden_dropout_prob=opt['hidden_dropout_prob'],
            layer_norm_eps=opt['layer_norm_eps'],
            pre_ln=opt.get('transformer_pre_ln', False),
        )

    def forward(self, 
            hidden_states: torch.Tensor, 
            attention_mask: Optional[torch.Tensor] = None, 
            head_mask: Optional[torch.Tensor] = None,
            **kwargs
        ) -> Tuple[torch.Tensor, torch.Tensor]:

        hidden_states, attention_probs, context = self.intra_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            **kwargs
        )
        hidden_states = self.ffn(hidden_states)
        
        return hidden_states, attention_probs, context


class DecoderLayer(nn.Module):
    def __init__(self, opt: Dict[str, Any], is_last=False):
        super(DecoderLayer, self).__init__()
        self.attr_layer_pos = opt.get('attr_layer_pos', 'cross2attr')
        assert self.attr_layer_pos in ['attr2cross', 'cross2attr', 'parallel']

        if opt.get('compositional_intra', False):
            intra_attention_class = CompositionalMHA
        else:
            intra_attention_class = MultiHeadAttention
        self.intra_attention = intra_attention_class(
            dim_hidden=opt['dim_hidden'],
            num_attention_heads=opt['num_attention_heads'],
            attention_probs_dropout_prob=opt['attention_probs_dropout_prob'],
            hidden_dropout_prob=opt['hidden_dropout_prob'],
            layer_norm_eps=opt['layer_norm_eps'],
            exclude_bias=opt.get('mha_exclude_bias', False),
            pre_ln=opt.get('transformer_pre_ln', False),
            have_relative_position_bias=opt.get('RPE', False),
            max_relative_position=opt.get('max_relative_position', None),
            attend_to_video=False,
            dim_semantic=opt.get('attribute_prediction_k', 500),
            dim_factor_scale=opt.get('dim_factor_scale', 2),
        )

        if opt.get('fusion', 'temporal_concat') == 'channel_concat':
            dim_key = dim_value = opt['dim_hidden'] * len(opt['modality'])
        else:
            dim_key = dim_value = opt['dim_hidden']

        modality = opt['modality'] if opt.get('modality_for_decoder', None) is None else opt['modality_for_decoder']
        hybrid_length = opt['n_frames'] * len(modality) + opt.get('use_attr_topk', 30)
        if opt['feats'] == 'SwinBERTDense' and  'm' in modality:
            hybrid_length = hybrid_length - opt['n_frames'] + 1568
        if 'r' in modality:
            hybrid_length += (opt['retrieval_topk'] - opt['n_frames'])

        if opt.get('compositional_inter', False):
            inter_attention_class = CompositionalMHA
        else:
            inter_attention_class = MultiHeadAttention
        self.inter_attention = inter_attention_class(
            dim_hidden=opt['dim_hidden'],
            dim_key=dim_key,
            dim_value=dim_value,
            num_attention_heads=opt['num_attention_heads'],
            attention_probs_dropout_prob=opt['attention_probs_dropout_prob'],
            hidden_dropout_prob=opt['hidden_dropout_prob'],
            layer_norm_eps=opt['layer_norm_eps'],
            exclude_bias=opt.get('mha_exclude_bias', False),
            pre_ln=opt.get('transformer_pre_ln', False),
            have_relative_position_bias=opt.get('RPE', False),
            max_relative_position=opt.get('max_relative_position', None),
            attend_to_video=True,
            has_ln=False if self.attr_layer_pos == 'parallel' else True,
            skip_connection=False if self.attr_layer_pos == 'parallel' else True,
            add_hybrid_attention_bias=opt.get('add_hybrid_attention_bias', False),
            hybrid_length=hybrid_length,
            dim_semantic=opt.get('attribute_prediction_k', 500),
            dim_factor_scale=opt.get('dim_factor_scale', 2),
        )

        if opt.get('use_attr', False):
            if 'att' in opt.get('use_attr_type', 'att'):
                self.attr_attention = copy.deepcopy(self.inter_attention)
        
        if self.attr_layer_pos == 'parallel':
            self.LayerNorm = nn.LayerNorm(opt['dim_hidden'], eps=opt['layer_norm_eps'])

        if opt.get('compositional_ffn', False):
            ffn_class = CompositionalFFN
        else:
            ffn_class = PositionwiseFeedForward
        self.ffn = ffn_class(
            dim_hidden=opt['dim_hidden'],
            dim_intermediate=opt['intermediate_size'],
            hidden_act=opt['hidden_act'],
            hidden_dropout_prob=opt['hidden_dropout_prob'],
            layer_norm_eps=opt['layer_norm_eps'],
            pre_ln=opt.get('transformer_pre_ln', False),
            dim_semantic=opt.get('attribute_prediction_k', 500),
            dim_factor_scale=opt.get('dim_factor_scale', 2),
        )
    
    def forward_attr_attention(self, 
            hidden_states: torch.Tensor, 
            head_mask: Optional[torch.Tensor] = None,
            **kwargs
        ):
        assert 'semantic_embs' in kwargs
        q = k = v = None

        return self.attr_attention(
            hidden_states=hidden_states,
            encoder_hidden_states=kwargs['semantic_embs'],
            attention_mask=None,
            head_mask=head_mask,
            attend_to_video=False,
            q=q, k=k, v=v
        )


    def forward(self, 
            hidden_states: torch.Tensor, 
            encoder_hidden_states: torch.Tensor, 
            attention_mask: Optional[torch.Tensor] = None, 
            encoder_attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            **kwargs
        ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        attention_probs = ()
        contexts = ()
        embs = ()
        
        hidden_states, intra_attention_probs, text_context = self.intra_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            attend_to_video=False,
            **kwargs
        )
        attention_probs = attention_probs + (intra_attention_probs, )
        contexts = contexts + (text_context, )
        embs = embs + (hidden_states.clone(), )

        if hasattr(self, 'attr_attention') and self.attr_layer_pos == 'attr2cross':
            hidden_states, attr_attention_probs, attr_context = self.forward_attr_attention(hidden_states, head_mask, **kwargs)
            if isinstance(attr_attention_probs, tuple):
                attention_probs = attention_probs + attr_attention_probs
            else:
                attention_probs = attention_probs + (attr_attention_probs, )
            contexts = contexts + (attr_context, )
            embs = embs + (hidden_states.clone(), )

        if hasattr(self, 'attr_attention') and self.attr_layer_pos == 'parallel':
            _, inter_attention_probs, inter_context = self.inter_attention(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                head_mask=head_mask,
                attend_to_video=True,
                **kwargs
            )

            _, attr_attention_probs, attr_context = self.forward_attr_attention(hidden_states, head_mask, **kwargs)

            hidden_states = self.LayerNorm(hidden_states + inter_context + attr_context)
            attention_probs = attention_probs + (inter_attention_probs, attr_attention_probs, )
            contexts = contexts + (inter_context, attr_context, )
            embs = embs + (hidden_states.clone(), )
        else:
            hidden_states, inter_attention_probs, context = self.inter_attention(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                head_mask=head_mask,
                attend_to_video=True,
                **kwargs
            )
            attention_probs = attention_probs + (inter_attention_probs, )
            contexts = contexts + (context, )
            embs = embs + (hidden_states.clone(), )

        if hasattr(self, 'attr_attention') and self.attr_layer_pos == 'cross2attr':
            hidden_states, attr_attention_probs, attr_context = self.forward_attr_attention(hidden_states, head_mask, **kwargs)
            if isinstance(attr_attention_probs, tuple):
                attention_probs = attention_probs + attr_attention_probs
            else:
                attention_probs = attention_probs + (attr_attention_probs, )
            contexts = contexts + (attr_context, )
            embs = embs + (hidden_states.clone(), )
            
        hidden_states = self.ffn(hidden_states, **kwargs)        
        return hidden_states, attention_probs, contexts, embs


class EncoderStack(nn.Module):
    def __init__(self, opt: Dict[str, Any], num_layers: int):
        super(EncoderStack, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(opt) for _ in range(num_layers)])
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        for layer in self.layers:
            hidden_states, *_ = layer(*args, **kwargs)
        return hidden_states


class CrossLayer(nn.Module):
    def __init__(self, opt: Dict[str, Any]):
        super(CrossLayer, self).__init__()
        if opt.get('fusion', 'temporal_concat') == 'channel_concat':
            dim_key = dim_value = opt['dim_hidden'] * len(opt['modality'])
        else:
            dim_key = dim_value = opt['dim_hidden']

        self.inter_attention = MultiHeadAttention(
            dim_hidden=opt['dim_hidden'],
            dim_key=dim_key,
            dim_value=dim_value,
            num_attention_heads=opt['num_attention_heads'],
            attention_probs_dropout_prob=opt['attention_probs_dropout_prob'],
            hidden_dropout_prob=opt['hidden_dropout_prob'],
            layer_norm_eps=opt['layer_norm_eps'],
            exclude_bias=opt.get('mha_exclude_bias', False),
            pre_ln=opt.get('transformer_pre_ln', False),
            have_relative_position_bias=opt.get('RPE', False),
            max_relative_position=opt.get('max_relative_position', None),
            attend_to_video=True
        )

        if not opt.get('crosslayer_no_ffn', False):
            self.ffn = PositionwiseFeedForward(
                dim_hidden=opt['dim_hidden'],
                dim_intermediate=opt['intermediate_size'],
                hidden_act=opt['hidden_act'],
                hidden_dropout_prob=opt['hidden_dropout_prob'],
                layer_norm_eps=opt['layer_norm_eps'],
                pre_ln=opt.get('transformer_pre_ln', False),
            )

    def forward(self, 
            hidden_states: torch.Tensor, 
            encoder_hidden_states: torch.Tensor, 
            attention_mask: Optional[torch.Tensor] = None, 
            encoder_attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            **kwargs
        ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        hidden_states, inter_attention_probs, context = self.inter_attention(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            attend_to_video=True,
            decoding_type='ARFormer',
            **kwargs
        )
        cross_embs = hidden_states.clone()

        if hasattr(self, 'ffn'):
            hidden_states = self.ffn(hidden_states)        
            
        return hidden_states, (None, inter_attention_probs), (None, context), (None, cross_embs)

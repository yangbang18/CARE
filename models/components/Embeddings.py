''' Define the common attention mechanisms '''

import torch
import torch.nn as nn
import math
from config import Constants
from typing import Dict, Any
import numpy as np


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super(PositionalEmbedding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class NaiveEmbeddings(nn.Module):
    """Construct the embeddings from word, position and category embeddings.
    """
    def __init__(self, 
        n_words: int, 
        n_positions: int, 
        dim_hidden: int,
        layer_norm_eps: float=1e-12,
        hidden_dropout_prob: float=0.5,
        padding_idx: int=None,
        prefix_len: int=0,
        suffix_len: int=0,
        has_ln: bool=True,
        has_dropout: bool=True,
    ):
        super(NaiveEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(n_words, dim_hidden, padding_idx=padding_idx) if n_words > 0 else None
        self.prefix_embeddings = nn.Embedding(prefix_len, dim_hidden) if prefix_len else None
        self.suffix_embeddings = nn.Embedding(suffix_len, dim_hidden) if suffix_len else None
        self.position_embeddings = nn.Embedding(n_positions, dim_hidden) if n_positions > 0 else None
        self.LayerNorm = nn.LayerNorm(dim_hidden, eps=layer_norm_eps) if has_ln else None
        self.dropout = nn.Dropout(hidden_dropout_prob) if has_dropout else None
    
    def forward(self, input_ids, input_embs=None):
        if input_embs is None:
            assert self.word_embeddings is not None
            input_embs = self.word_embeddings(input_ids)
        
        bsz = input_ids.size(0)

        dim = 1 if input_embs.dim() == 3 else 0
        if self.prefix_embeddings is not None:
            prefix_embs = self.prefix_embeddings.weight
            if input_embs.dim() == 3:
                prefix_embs = prefix_embs.unsqueeze(0).repeat(bsz, 1, 1)
            input_embs = torch.cat([prefix_embs, input_embs], dim=dim) # (bsz, prefix_len + seq_len, dim_hidden)
        
        if self.suffix_embeddings is not None:
            suffix_embs = self.suffix_embeddings.weight
            if input_embs.dim() == 3:
                suffix_embs = suffix_embs.unsqueeze(0).repeat(bsz, 1, 1)
            input_embs = torch.cat([input_embs, suffix_embs], dim=dim) # (bsz, seq_len + suffix_len, dim_hidden)
        
        if self.position_embeddings is not None:
            seq_length = input_embs.size(dim)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_embs.device)
            if input_embs.dim() == 3:
                position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)
            position_embeddings = self.position_embeddings(position_ids)
            input_embs = input_embs + position_embeddings

        if self.LayerNorm is not None:
            input_embs = self.LayerNorm(input_embs)
        
        if self.dropout is not None:
            input_embs = self.dropout(input_embs)
        
        return input_embs


class Embeddings(nn.Module):
    """Construct the embeddings from word, position and category embeddings.
    """
    def __init__(self, opt: Dict[str, Any]):
        super(Embeddings, self).__init__()
        if opt.get('pretrained_embs_path', ''):
            # the path to pretrained word embs is specified
            self.word_embeddings = nn.Embedding.from_pretrained(
                embeddings=torch.from_numpy(np.load(opt['pretrained_embs_path'])).float(),
                freeze=True,
            )
            assert self.word_embeddings.weight.shape[0] == opt['vocab_size']
            dim_word = self.word_embeddings.weight.shape[1]
            if dim_word != opt['dim_hidden']:
                self.w2h = nn.Linear(dim_word, opt['dim_hidden'], bias=False)
        else:
            self.word_embeddings = nn.Embedding(opt['vocab_size'], opt['dim_hidden'], padding_idx=Constants.PAD)
        
        self.trainable_pe = opt.get('trainable_pe', False)
        
        RPE = opt.get('RPE', False)
        RPE_keep_abs_pos = opt.get('RPE_keep_abs_pos', False)

        self.semantic_flag = 'emb' in opt.get('use_attr_type', '')
        self.prefix_flag = 'pp_emb' in opt.get('use_attr_type', '')

        if not RPE or (RPE and RPE_keep_abs_pos):
            if self.trainable_pe:
                self.position_embeddings = nn.Embedding(opt['max_len'], opt['dim_hidden'])
            else:
                self.position_embeddings = PositionalEmbedding(opt['max_len'], opt['dim_hidden'])

        self.with_category = opt.get('with_category', False)
        self.use_category_embs = opt.get('use_category_embs', False)
        if self.with_category:
            if self.use_category_embs:
                self.category_embeddings = nn.Linear(opt['dim_category'], opt['dim_hidden'])
            else:
                self.category_embeddings = nn.Embedding(opt['num_category'], opt['dim_hidden'])
            
        if not opt.get('transformer_pre_ln', False):
            self.LayerNorm = nn.LayerNorm(opt['dim_hidden'], eps=opt['layer_norm_eps'])
        self.dropout = nn.Dropout(opt['hidden_dropout_prob'])

    def forward(self, input_ids, category=None, additional_feats=None, only_word_and_position=False, **kwargs):
        embeddings = self.word_embeddings(input_ids)
        if hasattr(self, 'w2h'):
            embeddings = self.w2h(embeddings)

        if hasattr(self, 'position_embeddings'):
            if self.trainable_pe:
                if hasattr(self, 'position_embeddings_na'):
                    assert 'decoding_type_id' in kwargs
                    func = self.position_embeddings if kwargs['decoding_type_id'] == 0 else self.position_embeddings_na
                else:
                    func = self.position_embeddings
                bsz, seq_length = embeddings.shape[:2]
                position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
                position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)
                position_embeddings = func(position_ids)
            else:
                position_embeddings = self.position_embeddings(input_ids)

            embeddings = embeddings + position_embeddings
        
        if not only_word_and_position:
            if self.semantic_flag and self.prefix_flag:
                assert 'semantic_hidden_states' in kwargs
                embeddings = torch.cat([kwargs['semantic_hidden_states'].unsqueeze(1), embeddings], dim=1)
            
            if self.with_category:
                inputs = kwargs['category_embs'] if self.use_category_embs else category
                assert inputs is not None

                category_embeddings = self.category_embeddings(inputs.to(input_ids.device))
                if category_embeddings.dim() == 2:
                    category_embeddings = category_embeddings.unsqueeze(1) # [bsz, 1, dim_hidden]

                embeddings = embeddings + category_embeddings
            
            if additional_feats is not None:
                embeddings = embeddings + additional_feats
            
            if self.semantic_flag and not self.prefix_flag:
                assert 'semantic_hidden_states' in kwargs
                embeddings = embeddings + \
                    kwargs['semantic_hidden_states'].unsqueeze(1).expand_as(embeddings)
            
            if hasattr(self, 'decoding_type_embeddings'):
                assert 'decoding_type_id' in kwargs
                ids = input_ids.new(input_ids.shape[0], 1).fill_(kwargs['decoding_type_id'])
                embeddings = embeddings + self.decoding_type_embeddings(ids)
        
        if hasattr(self, 'LayerNorm'):
            embeddings = self.LayerNorm(embeddings)
        
        embeddings = self.dropout(embeddings)

        return (embeddings, )


class RelativePositionalEmbedding(nn.Module):
    def __init__(self, max_relative_position, num_attention_heads, attend_to_video):
        super().__init__()
        self.max_relative_position = max_relative_position
        self.num_attention_heads = num_attention_heads
        self.attend_to_video = attend_to_video
        self.embedding = nn.Embedding(max_relative_position * 2 + 1, num_attention_heads)

    def forward(self, length_q, length_k, device, bidirectional=True):
        range_vec_q = torch.arange(length_q, device=device, dtype=torch.long)
        range_vec_k = torch.arange(length_k, device=device, dtype=torch.long)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None] # shape (length_q, length_k)

        if self.attend_to_video:
            bidirectional = True

        distance_mat_clipped = torch.clamp(
            distance_mat, 
            -self.max_relative_position, 
            self.max_relative_position if bidirectional else 0
        )
        final_mat = distance_mat_clipped + self.max_relative_position # shape (length_q, length_k)

        # print(final_mat, bidirectional, length_q, length_k)

        values = self.embedding(final_mat)  # shape (length_q, length_k, num_attention_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_attention_heads, length_q, length_k)
        return values

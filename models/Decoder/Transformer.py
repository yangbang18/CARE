import torch
import torch.nn as nn
from typing import Dict
from config import Constants
from models.Predictor.pred_attribute import TextPostProcesser
from models.components.Embeddings import Embeddings
from models.components.Layers import DecoderLayer, EncoderStack, CrossLayer


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    if isinstance(seq_q, torch.Tensor):
        len_q = seq_q.size(1)
    else:
        assert type(seq_q) is int
        len_q = seq_q

    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


def get_subsequent_mask(seq, watch=0):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    if watch != 0 and len_s >= watch:
        assert watch > 0
        tmp = torch.tril(torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=-watch)
    else:
        tmp = None

    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    if tmp is not None:
        subsequent_mask += tmp
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask


def resampling(source, tgt_tokens):
    pad_mask = tgt_tokens.eq(Constants.PAD)
    length = (1 - pad_mask).sum(-1)
    bsz, seq_len = tgt_tokens.shape

    all_idx = []
    scale = source.size(1) / length.float()
    for i in range(bsz):
        idx = (torch.arange(0, seq_len, device=tgt_tokens.device).float() * scale[i].repeat(seq_len)).long()
        max_idx = tgt_tokens.new(seq_len).fill_(source.size(1) - 1)
        idx = torch.where(idx < source.size(1), idx, max_idx)
        all_idx.append(idx)
    all_idx = torch.stack(all_idx, dim=0).unsqueeze(2).repeat(1, 1, source.size(2))
    return source.gather(1, all_idx)


class TransformerDecoder(nn.Module):
    def __init__(self, opt, layer_func=DecoderLayer):
        super(TransformerDecoder, self).__init__()
        if opt.get('TAP_pos', False) or opt.get('TAP_ln', False):
            self.TPP = TextPostProcesser(opt)

        self.embedding = Embeddings(opt)
        self.layers = nn.ModuleList([
                layer_func(
                    opt, 
                    is_last=True if _ == opt['num_hidden_layers_decoder'] - 1 else False
                )for _ in range(opt['num_hidden_layers_decoder'])
            ])

        if opt.get('transformer_pre_ln', False):
            self.LayerNorm = nn.LayerNorm(opt['dim_hidden'], eps=opt['layer_norm_eps'])
        self.dropout = nn.Dropout(opt['hidden_dropout_prob'])
        
        self.enhance_input = opt['enhance_input']
        self.decoding_type = opt['decoding_type']
        self.opt = opt
        self.register_other_components(word_embeddings=self.get_word_embeddings())

    def _init_embedding(self, weight, option={}, is_numpy=False):
        if is_numpy:
            self.embedding.word_embeddings.weight.data = 0
        else:
            self.embedding.word_embeddings.weight.data.copy_(weight.data)
        if not option.get('train_emb', False):
            for p in self.embedding.word_embeddings.parameters():
                p.requires_grad = False

    def get_word_embeddings(self):
        return self.embedding.word_embeddings
    
    def set_word_embeddings(self, data: torch.FloatTensor):
        self.embedding.word_embeddings.weight.data = data
    
    def get_embeddings(self):
        return self.embedding
    
    def get_sentence_embeddings(self, input_ids, average_pooling=True):
        embs = self.embedding.word_embeddings(input_ids)
        if average_pooling:
            mask = embs.ne(Constants.PAD).float()
            n_words = mask.sum(dim=1, keepdim=True)
            embs = torch.sum(embs * mask.unsqueeze(2), dim=1) / n_words
        
        if hasattr(self, 'TPP'):
            embs = self.TPP(embs)
        return embs
    
    def get_attr_embeddings(self, attr_input_ids):
        attr_embs = self.embedding.word_embeddings(attr_input_ids)
        if hasattr(self, 'TPP'):
            attr_embs = self.TPP(attr_embs)
        return attr_embs

    def register_other_components(self, **kwargs):
        pass
    
    def run_other_components(self, context, hidden_states, input_embeddings, **kwargs) -> Dict[str, torch.Tensor]:
        outputs = {}
        return outputs
    
    def postprocess_attention_mask(self, attention_mask):
        if self.opt.get('use_attr', False) \
            and ('prefix' in self.opt.get('use_attr_type', '') or 'pp' in self.opt.get('use_attr_type', '')):
            if 'prefix' in self.opt['use_attr_type']:
                prefix_len = self.opt['use_attr_topk']
                # prefix_attention_mask_left = torch.zeros(bsz, seq_len, prefix_len).to(attention_mask.device)
                # prefix_attention_mask_top = torch.zeros(bsz, prefix_len, seq_len+prefix_len).to(attention_mask.device)
                # prefix_attention_mask_top[:, :, prefix_len:] = 1
            else:
                # global semantic guidance
                prefix_len = 1

            bsz, seq_len = attention_mask.shape[:2]
            # positions with zero values mean that they can be attended to
            prefix_attention_mask_left = torch.zeros(bsz, seq_len, prefix_len).to(attention_mask.device) # each word can observe all semantic signals
            prefix_attention_mask_top = torch.ones(bsz, prefix_len, seq_len+prefix_len).to(attention_mask.device)
            prefix_attention_mask_top[:, torch.arange(prefix_len), torch.arange(prefix_len)] = 0 # the semantic siginal can only observe itself
            
            attention_mask = torch.cat([prefix_attention_mask_left, attention_mask], dim=2)
            attention_mask = torch.cat([prefix_attention_mask_top, attention_mask], dim=1).bool()

        return attention_mask
    
    def postprocess_input_embs(self, input_embs, semantic_embs=None, **kwargs):
        if self.opt.get('use_attr', False) and 'prefix' in self.opt.get('use_attr_type', ''):
            assert semantic_embs is not None
            return torch.cat([semantic_embs, input_embs], dim=1)

        return input_embs

    def forward(self, input_ids, encoder_hidden_states=None, category=None, head_mask=None, return_input_embs=False, **kwargs):
        decoding_type = kwargs.pop('decoding_type', self.decoding_type)

        if isinstance(encoder_hidden_states, list):
            assert len(encoder_hidden_states) == 1
            encoder_hidden_states = encoder_hidden_states[0]
        
        # get intra-attention (self-attention) mask
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=input_ids, seq_q=input_ids)
        if decoding_type == 'NARFormer':
            attention_mask = slf_attn_mask_keypad
        else:
            slf_attn_mask_subseq = get_subsequent_mask(input_ids)
            attention_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        
        attention_mask = self.postprocess_attention_mask(attention_mask)

        # get inter-attention (cross-modal attention) mask
        src_seq = torch.ones(encoder_hidden_states.size(0), encoder_hidden_states.size(1)).to(encoder_hidden_states.device)
        encoder_attention_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=attention_mask.size(1))

        additional_feats = None
        if decoding_type == 'NARFormer':
            if self.enhance_input == 0:
                pass
            elif self.enhance_input == 1:
                additional_feats = resampling(encoder_hidden_states, input_ids)
            elif self.enhance_input == 2:
                additional_feats = encoder_hidden_states.mean(1).unsqueeze(1).repeat(1, input_ids.size(1), 1)
            else:
                raise ValueError('enhance_input shoud be either 0, 1 or 2')

        input_embs, *other_embs = self.embedding(input_ids, additional_feats=additional_feats, category=category, **kwargs)
        if hasattr(self, 'post_embedding'):
            input_embs = self.post_embedding(input_embs, attention_mask=attention_mask)
        
        original_input_embs = input_embs.clone()
        input_embs = self.postprocess_input_embs(input_embs, **kwargs)

        if return_input_embs:
            return input_embs

        all_hidden_states = [input_embs]
        all_intra_attentions = ()
        all_inter_attentions = ()
        all_attr_attentions = ()
        all_gate_probs = ()

        for layer in self.layers:
            hidden_states, attention_probs, contexts, embs = layer(
                hidden_states=all_hidden_states[-1], 
                encoder_hidden_states=encoder_hidden_states, 
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask, 
                head_mask=head_mask,
                decoding_type=decoding_type,
                n_frames=self.opt['n_frames'],
                **kwargs
            )

            intra_attention_probs, inter_attention_probs, *__ = attention_probs
            text_context, context, *_ = contexts
            self_embs, cross_embs, *_ = embs

            all_hidden_states.append(hidden_states)
            all_intra_attentions = all_intra_attentions + (intra_attention_probs, )
            all_inter_attentions = all_inter_attentions + (inter_attention_probs, )
            if len(__):
                all_attr_attentions = all_attr_attentions + (__[0], )
            if len(__) == 2:
                all_gate_probs = all_gate_probs + (__[1], )

        hidden_states = all_hidden_states[-1]

        if hasattr(self, 'LayerNorm'):
            hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        main_outputs = {
            'hidden_states': hidden_states,
            'all_hidden_states': all_hidden_states,
            'all_intra_attentions': all_intra_attentions,
            'all_inter_attentions': all_inter_attentions,
            'attention_probs': all_inter_attentions[-1].mean(1),
            'context': context,
            'text_context': text_context,
            'self_embs': self_embs,
            'cross_embs': cross_embs,
            'input_embs': original_input_embs,
            'input_embs_exclude_bos': original_input_embs[:, 1:, :],
            'sentence_embs': self.get_sentence_embeddings(input_ids, average_pooling=False)
        }

        if self.opt.get('use_attr'):
            main_outputs['attr_attention_probs'] = all_attr_attentions # (bsz, n_heads, max_len-1, n_topk+2)
            main_outputs['gate_probs'] = all_gate_probs

        attr_input_ids = kwargs.get('attr_input_ids', None)
        if attr_input_ids is not None:
            main_outputs['attr_embs'] = self.get_attr_embeddings(attr_input_ids)

        other_components_outputs = self.run_other_components(
            context=main_outputs['context'], 
            hidden_states=main_outputs['hidden_states'],
            input_embeddings=main_outputs['all_hidden_states'][0],
            **kwargs
        )
        return {**main_outputs, **other_components_outputs}


class TwoStageTransformerDecoder(TransformerDecoder):
    def forward(self, input_ids, *args, **kwargs):
        if isinstance(input_ids, list):
            assert len(input_ids) == 2 or len(input_ids) == 3
            outputs1 = super().forward(input_ids[0], *args, **kwargs) # all [mask] for the generation of coarse grained templates 
            outputs2 = super().forward(input_ids[1], *args, **kwargs) # masked language modeling
            outputs2['hidden_states'] = [outputs1['hidden_states'], outputs2['hidden_states']]

            if len(input_ids) == 3:
                outputs2['input_embs'] = super().forward(input_ids[2], *args, **kwargs, return_input_embs=True)
                outputs2['sentence_embs'] = self.get_sentence_embeddings(input_ids[2], average_pooling=False)

            return outputs2
        else:
            assert not self.training
            return super().forward(input_ids, *args, **kwargs)

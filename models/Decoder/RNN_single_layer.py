import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from typing import Dict
from config import Constants
from models.components.Attention import AdditiveAttention, MultiLevelAttention
from models.components.SubLayers import MultiHeadAttention


class RNNDecoderBase(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.rnn_type = opt.get('rnn_type', 'lstm').lower()
        self.with_category = opt.get('with_category', False)
        self.use_category_embs = opt.get('use_category_embs', False)
        if self.with_category:
            if self.use_category_embs:
                self.dim_category = opt.get('dim_category', 300)
            else:
                self.dim_category = opt.get('num_category', 20)
        else:
            self.dim_category = 0
        
        self.hidden_size = opt['dim_hidden']

        self.num_modality = len(opt['modality_for_decoder']) if opt.get('modality_for_decoder', None) else len(opt['modality'])
        self.num_att_modality = 1 if opt['fusion'] == 'temporal_concat' else self.num_modality

        # concept-aware params
        self.semantic_global_flag, self.semantic_local_flag = False, False
        if opt.get('use_attr', False):
            self.semantic_global_flag = 'emb' in opt.get('use_attr_type', '')
            self.semantic_local_flag = 'att' in opt.get('use_attr_type', '')

    def _init_lstm_forget_bias(self, forget_bias=1.0): 
        # TODO: make it more elegant 
        print("====> initializing the forget bias of LSTM to %g" % forget_bias)
        
        for name, module in self.named_children():
            if isinstance(module, nn.LSTMCell):
                ingate, forgetgate, cellgate, outgate = module.bias_ih.chunk(4, 0)
                forgetgate.data.add_(forget_bias)
                module.bias_ih = nn.Parameter(torch.cat([ingate, forgetgate, cellgate, outgate], dim=0))

                ingate, forgetgate, cellgate, outgate = module.bias_hh.chunk(4, 0)
                forgetgate.data.add_(forget_bias)
                module.bias_hh = nn.Parameter(torch.cat([ingate, forgetgate, cellgate, outgate], dim=0))

    def register_other_components(self, **kwargs):
        pass
    
    def run_other_components(self, context, input_ids, hidden_states, **kwargs) -> Dict[str, torch.Tensor]:
        outputs = {}
        return outputs

    def prepare_word_embeddings(self):
        if self.opt.get('pretrained_embs_path', ''):
            # the path to pretrained word embs is specified
            module = nn.Embedding.from_pretrained(
                embeddings=torch.from_numpy(np.load(self.opt['pretrained_embs_path'])).float(),
                freeze=True,
                padding_idx=Constants.PAD,
            )
            assert module.weight.shape[0] == self.opt['vocab_size']    
        else:
            module = nn.Embedding(self.opt['vocab_size'], self.opt['dim_hidden'], padding_idx=Constants.PAD)
        return module

    def get_word_embeddings(self):
        return self.embedding

    def get_embeddings(self):
        return self.embedding

    def set_word_embeddings(self, data):
        raise NotImplementedError('Not support right now!')
    
    def get_sentence_embeddings(self, input_ids, average_pooling=True):
        embs = self.embedding(input_ids)
        if average_pooling:
            mask = embs.ne(Constants.PAD).float()
            n_words = mask.sum(dim=1, keepdim=True)
            embs = torch.sum(embs * mask.unsqueeze(2), dim=1) / n_words
        
        if hasattr(self, 'TPP'):
            embs = self.TPP(embs)
        return embs

    def init_decoder_rnn_hidden_states(self, encoder_hidden_states, **kwargs):
        assert hasattr(self, 'rnn_type'), 'Please make sure the derived classes have `self.rnn_type` (str, `lstm` or `gru`)'
        
        bsz = encoder_hidden_states[0].size(0) if isinstance(encoder_hidden_states, list) else encoder_hidden_states.size(0)
        weight = next(self.parameters())

        if hasattr(self, 'v2h'):
            if self.v2h is None:
                hidden = self.get_mean_video_features(encoder_hidden_states)
            else:
                hidden = self.v2h(self.get_mean_video_features(encoder_hidden_states))
        else:
            hidden = weight.new_zeros([bsz, self.hidden_size])

        if self.rnn_type == 'lstm':
            if hasattr(self, 'v2c'):
                if self.v2c is None:
                    cell = self.get_mean_video_features(encoder_hidden_states)
                else:
                    cell = self.v2c(self.get_mean_video_features(encoder_hidden_states))
            else:
                cell = weight.new_zeros([bsz, self.hidden_size])
                
            decoder_rnn_hidden_states = (hidden, cell)
        else:
            decoder_rnn_hidden_states = hidden

        if hasattr(self, 'init_decoder_rnn_hidden_states_post_processing'):
            return self.init_decoder_rnn_hidden_states_post_processing(decoder_rnn_hidden_states)
        
        return decoder_rnn_hidden_states

    def preparation_before_feedforward(self, decoder_rnn_hidden_states, encoder_hidden_states, **kwargs):
        if decoder_rnn_hidden_states is None:
            decoder_rnn_hidden_states = self.init_decoder_rnn_hidden_states(encoder_hidden_states, **kwargs)

        return decoder_rnn_hidden_states, encoder_hidden_states

    def get_mean_video_features(self, encoder_hidden_states):
        if not isinstance(encoder_hidden_states, list):
            encoder_hidden_states = [encoder_hidden_states]

        mean_v = torch.stack(encoder_hidden_states, dim=0).mean(0)
        mean_v = mean_v.mean(1) # [bsz, dim_hidden]
        return mean_v

    def get_hidden_states(self, decoder_rnn_hidden_states):
        # the function `init_decoder_rnn_hidden_states` has confirmed hasattr(self, 'rnn_type')
        if self.rnn_type == 'lstm':
            hidden_states = decoder_rnn_hidden_states[0]
        else:
            hidden_states = decoder_rnn_hidden_states
        
        if len(hidden_states.shape) == 3:
            assert hidden_states.size(0) == 1
            hidden_states = hidden_states.squeeze(0)

        return hidden_states
    
    def add_auxiliary_info_to_inputs(self, inputs, category=None, **kwargs):
        if not isinstance(inputs, list):
            inputs = [inputs]

        if self.with_category and self.use_category_embs:
            other_inputs = [kwargs['category_embs']]
        elif self.with_category:
            other_inputs = [category]
        else:
            other_inputs = []

        return inputs + other_inputs
    
    def scheduled(self, i, sample_mask, item, prob_prev):
        if item is None or prob_prev is None:
            return None
        if sample_mask.sum() == 0:
            it = item[:, i].clone()
        else:
            sample_ind = sample_mask.nonzero().view(-1)
            it = item[:, i].data.clone()
            prob_prev = prob_prev.detach() # fetch prev distribution: shape Nx(M+1)
            it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))

        return it

    def forward_step(self, it, encoder_hidden_states, decoder_rnn_hidden_states=None, **kwargs):
        raise NotImplementedError('Please implement `forward_step` in the derived classes')
    
    def forward(self, input_ids, encoder_hidden_states, **kwargs):
        # teacher forcing loop, i.e., without schedule sampling
        # schedule sampling is implemented in the class `RNNSeq2Seq` in models/Framework.py
        assert input_ids.dim() == 2, "(bsz, seq_len)"

        hidden_states = []
        attention_probs = []
        logits = []
        other_component_outputs = defaultdict(list)

        schedule_sampling_prob = 0 if not self.training else kwargs.get('schedule_sampling_prob', 0)
        cls_head = kwargs['cls_head']

        decoder_rnn_hidden_states = None
        for i in range(input_ids.size(1)):
            if i >= 1 and schedule_sampling_prob > 0:
                # replace gt words with model's predictions with a certain probability (`schedule_sampling_prob`)
                prob = input_ids.new(input_ids.size(0)).float().uniform_(0, 1) # `prob` locates in the same device as input_ids
                mask = prob < schedule_sampling_prob
                it = self.scheduled(i, mask, input_ids, prob_prev=torch.softmax(logits[-1], dim=-1))
            else:
                # teacher forcing
                it = input_ids[:, i]

            decoding_phase_outputs = self.forward_step(it, encoder_hidden_states, decoder_rnn_hidden_states, **kwargs)

            decoder_rnn_hidden_states = decoding_phase_outputs.pop('decoder_rnn_hidden_states')
            hidden_states.append(decoding_phase_outputs.pop('hidden_states'))
            attention_probs.append(decoding_phase_outputs.pop('attention_probs'))
            logits.append(cls_head(hidden_states[-1]))

            for k, v in decoding_phase_outputs.items():
                other_component_outputs[k].append(v)

        for k in other_component_outputs:
            other_component_outputs[k] = torch.stack(other_component_outputs[k], dim=1)
        
        return {
            'hidden_states': torch.stack(hidden_states, dim=1), # [bsz, max_len-1, dim_hidden]
            'attention_probs': torch.stack(attention_probs, dim=2), # [bsz, num_feats, max_len-1, n_frames]
            'logits': torch.stack(logits, dim=1), # [bsz, max_len-1, vocab_size]
            'sentence_embs': self.get_sentence_embeddings(input_ids, average_pooling=False),
            **other_component_outputs
        }


class SingleLayerRNNDecoder(RNNDecoderBase):
    def __init__(self, opt, has_v2h_v2c=True):
        super().__init__(opt)
        # define the word embeddings
        self.embedding = self.prepare_word_embeddings()
        dim_word = self.embedding.weight.shape[1]
        self.LayerNorm = nn.LayerNorm(opt['dim_hidden'], eps=opt.get('layer_norm_eps', 1e-12))

        # define the rnn module
        self.rnn_type = opt.get('rnn_type', 'lstm').lower()
        rnn_func = nn.LSTMCell if self.rnn_type == 'lstm' else nn.GRUCell

        self.rnn = rnn_func(
            # inputs: [y(t-1); category (optional); att(feats); semantic_att (optional)]
            input_size=dim_word \
                + opt['dim_hidden'] * (self.num_att_modality + 1 * self.semantic_local_flag) \
                    + self.dim_category,
            hidden_size=opt['dim_hidden']
        )

        if has_v2h_v2c:
            self.v2h = nn.Linear(opt['dim_hidden'], opt['dim_hidden']) # to init h0
            if self.rnn_type == 'lstm':
                self.v2c = nn.Linear(opt['dim_hidden'], opt['dim_hidden']) # to init c0
        else:
            self.v2h = self.v2c = None

        # define the attention module
        hybrid_length = opt['n_frames'] * self.num_modality + opt.get('use_attr_topk', 30)

        self.mha_flag = opt.get('rnn_use_mha', False)
        if self.mha_flag:
            self.att = MultiHeadAttention(
                dim_hidden=opt['dim_hidden'],
                num_attention_heads=opt['num_attention_heads'],
                attention_probs_dropout_prob=opt['attention_probs_dropout_prob'],
                hidden_dropout_prob=opt['hidden_dropout_prob'],
                layer_norm_eps=opt['layer_norm_eps'],
                attend_to_video=True,
                add_hybrid_attention_bias=opt.get('add_hybrid_attention_bias', False),
                hybrid_length=hybrid_length,
            )
        else:
            att_func = MultiLevelAttention if opt.get('with_multileval_attention', False) \
                else AdditiveAttention

            self.att = att_func(
                dim_hidden=opt['dim_hidden'],
                dim_feats=[opt['dim_hidden']] * self.num_att_modality,
                dim_mid=opt['dim_hidden'],
                feats_share_weights=opt.get('feats_share_weights', False),
                add_hybrid_attention_bias=opt.get('add_hybrid_attention_bias', False),
                hybrid_length=hybrid_length,
            )

        if self.semantic_local_flag:
            self.semantic_att = AdditiveAttention(
                dim_hidden=opt['dim_hidden'],
                dim_feats=opt['dim_hidden'],
                dim_mid=opt['dim_hidden'],
            )

        self.dropout = nn.Dropout(opt['hidden_dropout_prob'])
        self.register_other_components(word_embeddings=self.get_word_embeddings())
        self._init_lstm_forget_bias()

    def forward_step(self, it, encoder_hidden_states, decoder_rnn_hidden_states=None, **kwargs):
        assert it.dim() == 1, '(bsz, )'

        decoder_rnn_hidden_states, encoder_hidden_states = self.preparation_before_feedforward(
            decoder_rnn_hidden_states, encoder_hidden_states, **kwargs)

        # attend to encoder's outputs
        if self.mha_flag:
            context, attention_probs, _ = self.att(
                hidden_states=self.get_hidden_states(decoder_rnn_hidden_states).unsqueeze(1), # use h(t-1) as the query
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=None,
                attend_to_video=True,
                **kwargs
            )
            context = context.squeeze(1)
        else:
            context, attention_probs = self.att(
                hidden_states=self.get_hidden_states(decoder_rnn_hidden_states), # use h(t-1) as the query
                feats=encoder_hidden_states,
                **kwargs,
            )

        category = kwargs.get('category', None)
        rnn_inputs = [self.embedding(it)] + ([category] if self.with_category else []) + [context]
        main_outputs = {'context': context, 'attention_probs': attention_probs}

        if self.semantic_global_flag:
            assert 'semantic_hidden_states' in kwargs
            rnn_inputs[0] = rnn_inputs[0] + kwargs['semantic_hidden_states']
        
        # apply layer normalization on word embeddings
        rnn_inputs[0] = self.LayerNorm(rnn_inputs[0])

        # attend to concept embs
        if self.semantic_local_flag:
            assert 'semantic_embs' in kwargs and kwargs['semantic_embs'] is not None
            semantic_context, semantic_attention_probs = self.semantic_att(
                hidden_states=self.get_hidden_states(decoder_rnn_hidden_states), # use h(t-1) as the query
                feats=kwargs['semantic_embs'],
            )
            
            rnn_inputs.append(semantic_context)
            main_outputs['semantic_attention_probs'] = semantic_attention_probs
        
        rnn_inputs = self.dropout(torch.cat(rnn_inputs, dim=-1))
        decoder_rnn_hidden_states = self.rnn(rnn_inputs, decoder_rnn_hidden_states)

        final_hidden_states = self.get_hidden_states(decoder_rnn_hidden_states) # get h(t)

        main_outputs.update({
            'hidden_states': self.dropout(final_hidden_states),
            'decoder_rnn_hidden_states': decoder_rnn_hidden_states,
        })

        other_components_outputs = self.run_other_components(
            context=context, 
            input_ids=it, 
            hidden_states=final_hidden_states
        )
        return {**main_outputs, **other_components_outputs}


class VOERNNDecoder(SingleLayerRNNDecoder):
    def __init__(self, opt):
        super().__init__(opt, has_v2h_v2c=False)

import torch
import torch.nn as nn
from models.components.Attention import AdditiveAttention, MultiLevelAttention
from models.components.SubLayers import MultiHeadAttention
from models.Decoder.RNN_single_layer import RNNDecoderBase


class TwoLayerRNNDecoderBase(RNNDecoderBase):
    def forward_step(self, it, encoder_hidden_states, decoder_rnn_hidden_states=None, **kwargs):
        assert it.dim() == 1, '(bsz, )'
        assert hasattr(self, 'bottom_rnn')
        assert hasattr(self, 'top_rnn')

        decoder_rnn_hidden_states, encoder_hidden_states = self.preparation_before_feedforward(
            decoder_rnn_hidden_states, encoder_hidden_states, **kwargs)

        bottom_rnn_inputs, other_outputs_bottom = self.prepare_bottom_rnn_inputs(
            it, decoder_rnn_hidden_states, encoder_hidden_states, **kwargs)
        decoder_rnn_hidden_states[0] = self.bottom_rnn(bottom_rnn_inputs, decoder_rnn_hidden_states[0])

        top_rnn_inputs, other_outputs_top = self.prepare_top_rnn_inputs(
            decoder_rnn_hidden_states, encoder_hidden_states, **kwargs)
        decoder_rnn_hidden_states[1] = self.top_rnn(top_rnn_inputs, decoder_rnn_hidden_states[1])

        final_hidden_states, other_outputs_final = self.prepare_final_hidden_states(
            decoder_rnn_hidden_states, **other_outputs_bottom, **other_outputs_top)

        return {
            'hidden_states': self.dropout(final_hidden_states),
            'decoder_rnn_hidden_states': decoder_rnn_hidden_states,
            'input_embs_bottom': self.get_hidden_states(decoder_rnn_hidden_states[0]),
            **other_outputs_bottom,
            **other_outputs_top,
            **other_outputs_final
        }
    
    def init_decoder_rnn_hidden_states_post_processing(self, decoder_rnn_hidden_states):
        if self.rnn_type == 'lstm':
            hidden, cell = decoder_rnn_hidden_states
            return [
                decoder_rnn_hidden_states,
                (hidden.new_zeros(*hidden.shape), hidden.new_zeros(*hidden.shape))
            ]
        else:
            return [
                decoder_rnn_hidden_states, 
                decoder_rnn_hidden_states.new_zeros(*decoder_rnn_hidden_states.shape)
            ]

    def prepare_bottom_rnn_inputs(self, it, decoder_rnn_hidden_states, encoder_hidden_states, **kwargs):
        raise NotImplementedError('Please implement the `prepare_bottom_rnn_inputs` function in derived classes')

    def prepare_top_rnn_inputs(self, decoder_rnn_hidden_states, encoder_hidden_states, **kwargs):
        raise NotImplementedError('Please implement the `prepare_top_rnn_inputs` function in derived classes')

    def prepare_final_hidden_states(self, decoder_rnn_hidden_states, **kwargs):
        raise NotImplementedError('Please implement the `prepare_final_hidden_states` function in derived classes')


class TopDownAttentionRNNDecoder(TwoLayerRNNDecoderBase):
    ''' Reproduce the decoder of `Bottom-Up and Top-Down Attention
        for Image Captioning and Visual Question Answering` (CVPR 2018)
        https://arxiv.org/pdf/1707.07998.pdf
    '''
    def __init__(self, opt):
        super().__init__(opt)
        # define the word embeddings
        self.embedding = self.prepare_word_embeddings()
        dim_word = self.embedding.weight.shape[1]
        self.LayerNorm = nn.LayerNorm(opt['dim_hidden'], eps=opt.get('layer_norm_eps', 1e-12))

        # define the rnn module
        self.rnn_type = opt.get('rnn_type', 'lstm').lower()
        rnn_func = nn.LSTMCell if self.rnn_type == 'lstm' else nn.GRUCell

        self.bottom_rnn = rnn_func(
            # inputs: [y(t-1); top_h(t-1); mean_v; category (optional)]
            input_size=dim_word + opt['dim_hidden'] * 2 + self.dim_category,
            hidden_size=opt['dim_hidden']
        )
        self.top_rnn = rnn_func(
            # inputs: [bottom_h(t); att(feats); semantic_att (optional)]
            input_size=opt['dim_hidden'] + opt['dim_hidden'] * (self.num_att_modality + 1 * self.semantic_local_flag),
            hidden_size=opt['dim_hidden']
        )

        self.v2h = nn.Sequential(nn.Linear(opt['dim_hidden'], opt['dim_hidden']), nn.Tanh())
        self.v2c = nn.Sequential(nn.Linear(opt['dim_hidden'], opt['dim_hidden']), nn.Tanh())

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
        self._init_lstm_forget_bias()

    def prepare_bottom_rnn_inputs(self, it, decoder_rnn_hidden_states, encoder_hidden_states, **kwargs):
        inputs = [
            self.embedding(it),
            self.get_hidden_states(decoder_rnn_hidden_states[1]), # top_h(t-1)
            self.get_mean_video_features(encoder_hidden_states),
        ]

        if self.semantic_global_flag:
            assert 'semantic_hidden_states' in kwargs
            inputs[0] = inputs[0] + kwargs['semantic_hidden_states']
        
        # apply layer normalization on word embeddings
        inputs[0] = self.LayerNorm(inputs[0])
        
        inputs = self.add_auxiliary_info_to_inputs(inputs, **kwargs)

        inputs = self.dropout(torch.cat(inputs, dim=-1))
        return inputs, {'input_embs': self.embedding(it)}

    def prepare_top_rnn_inputs(self, decoder_rnn_hidden_states, encoder_hidden_states, **kwargs):
        attention_rnn_hidden_states = self.get_hidden_states(decoder_rnn_hidden_states[0]) # bottom_h(t)

        if self.mha_flag:
            context, attention_probs, _ = self.att(
                hidden_states=attention_rnn_hidden_states.unsqueeze(1), # bottom_h(t) as the query
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=None,
                attend_to_video=True,
                **kwargs
            )
            context = context.squeeze(1)
        else:
            context, attention_probs = self.att(
                hidden_states=attention_rnn_hidden_states, # bottom_h(t) as the query
                feats=encoder_hidden_states,
                **kwargs
            )

        inputs = [attention_rnn_hidden_states, context]
        output_dict = {'attention_probs': attention_probs, 'context': context}

        if self.semantic_local_flag:
            assert 'semantic_embs' in kwargs and kwargs['semantic_embs'] is not None
            semantic_context, semantic_attention_probs = self.semantic_att(
                hidden_states=attention_rnn_hidden_states, # bottom_h(t) as the query
                feats=kwargs['semantic_embs'],
            )
            
            inputs.append(semantic_context)
            output_dict['semantic_attention_probs'] = semantic_attention_probs

        inputs = self.dropout(torch.cat(inputs, dim=-1))
        return inputs, output_dict

    def prepare_final_hidden_states(self, decoder_rnn_hidden_states, **kwargs):
        return self.get_hidden_states(decoder_rnn_hidden_states[1]), {}


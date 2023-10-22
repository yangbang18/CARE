import torch
import torch.nn as nn

from typing import List, Dict, Any, Optional

from .Backbone import get_backbone
from .Encoder import get_encoder
from .Predictor import get_predictor
from .Decoder import get_decoder 
from .Head import get_cls_head
from .Pointer import get_pointer


def get_framework(opt: Dict[str, Any]) -> nn.Module:
    if 'rnn' in opt['decoder'].lower():
        seq2seq_class = RNNSeq2Seq
    else:
        seq2seq_class = TransformerSeq2Seq

    # see the explaination of `input_keys_for_decoder` in the class `Seq2SeqBase` below
    input_keys_for_decoder = ['encoder_hidden_states']
    
    if opt.get('with_category', False):
        if opt.get('use_category_embs', False):
            input_keys_for_decoder += ['category_embs']
        else:
            input_keys_for_decoder += ['category']
    
    if opt.get('use_attr', False) and ('prefix' in opt['use_attr_type'] or 'att' in opt['use_attr_type'].lower()):
        input_keys_for_decoder += ['semantic_embs']
    
    if 'emb' in opt.get('use_attr_type', ''):
        input_keys_for_decoder += ['semantic_hidden_states']
    
    if opt.get('compositional_intra', False) or opt.get('compositional_inter', False) or opt.get('compositional_ffn'):
        input_keys_for_decoder += ['preds_attr']
    
    if opt.get('pointer'):
        input_keys_for_decoder += ['ret_text_embs', 'ret_input_ids']
        assert seq2seq_class is TransformerSeq2Seq
    
    return seq2seq_class(
        backbone=get_backbone(opt),
        encoder=get_encoder(opt),
        predictor=get_predictor(opt),
        decoder=get_decoder(opt),
        pointer=get_pointer(opt),
        cls_head=get_cls_head(opt),
        input_keys_for_decoder=input_keys_for_decoder,
        opt=opt,
    )


class Seq2SeqBase(nn.Module):
    def __init__(self,
            backbone: Optional[nn.Module],
            encoder: nn.Module,
            predictor: Optional[nn.Module],
            decoder: nn.Module,
            pointer: nn.Module,
            cls_head: nn.Module,
            input_keys_for_decoder: List[str] = ['encoder_hidden_states'],
            opt: Dict[str, Any] = {},
            init_weights: bool = True
        ):
        super().__init__()
        ''' 
            Overview of the framework:
                backone --> encoder --> decoder --> cls_head
                                |          ^
                                |          |
                                ----> predictor             
            args:
                backbone:   process raw video frames and encode them to a sequence of features
                encoder:    further encodes features from the `backbone` to more compact ones
                predictor:  to complement some auxiliary tasks,
                            e.g., predicting the sequence length based on the outputs of the encoder
                decoder:    yields hidden states given previsouly generated tokens 
                            and the outputs of the encoder and predictor
                cls_head:   maps the hidden states to logits over the vocabulary
                input_keys_for_decoder: see the explanation below
        '''
        self.backbone = backbone
        self.encoder = encoder
        self.predictor = predictor
        self.decoder = decoder
        self.pointer = pointer
        self.cls_head = cls_head

        # For convenience, we group decoder inputs into two subgroups
        #   1) previously generated sequences (`input_ids`), which will dynamically change during inference
        #   2) other information like the outputs of the encoder (`encoder_hidden_states`),
        #      auxiliary category information (`category`, optional) etc, which will be 
        #      expanded (repetaed) first and then fixed during inference
        # Herein, we define `input_keys_for_decoder` to specify the inputs of the 2nd subgroup for flexibility
        self.input_keys_for_decoder = input_keys_for_decoder
        '''example (pesudo code from models/Translator.py):

            # before defining `input_keys_for_decoder`
            encoder_hidden_states = auto_enlarge(encoding_phase_outputs['encoder_hidden_states'], beam_size)
            category = auto_enlarge(batch['category'], beam_size)
            some_other_inputs = auto_enlarge(some_other_inputs, beam_size)
            beam_decode_step(..., inst_dec_beams, ..., encoder_hidden_states, category, some_other_inputs, ...)

            # after defining `input_keys_for_decoder`
            inputs_for_decoder = model.prepare_inputs_for_decoder(encoding_phase_outputs, batch)
            for key in inputs_for_decoder:
                inputs_for_decoder[key] = auto_enlarge(inputs_for_decoder[key], beam_size)
            beam_decode_step(..., inst_dec_beams, ..., inputs_for_decoder, ...)
        '''
        self.opt = opt
        if init_weights:
            self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            if 'backbone' in name:
                # skip the pretrained weights of the backbone
                continue
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                if self.opt.get('skip_word_embeddings', False):
                    print('!!!! skip word_embeddings')
                    continue
                if module.weight.requires_grad:
                    nn.init.xavier_uniform_(module.weight)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.weight.data.fill_(1.0)
                module.bias.data.zero_()
    
    def get_keys_to_device(self, teacher_forcing=False, **kwargs):
        # if we do not use pytorch_lightning.Traniner to automatically manage the device of data
        # we must know which part of data should be moved to specified device
        keys = ['feats']

        # if teacher_forcing:
        keys.append('input_ids')

        for k in self.input_keys_for_decoder:
            # exclude intermediate hidden states
            if 'hidden_states' not in k:
                keys.append(k)
        return keys
    
    def encoding_phase(self, feats: List[torch.Tensor], **kwargs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        feats, other_feats = feats[:len(self.opt['modality'])], feats[len(self.opt['modality']):]
        
        semantic_logits = None
        text_embs = None
        if other_feats:
            if self.opt.get('logits', []):
                semantic_logits = other_feats[0]
            if self.opt.get('retrieval', False):
                text_embs = other_feats[-1]

        if self.backbone is not None and not kwargs.get('skip_backbone', False):
            # some items in the `feats` may be frames or snippets,
            # which will be encoded to features by the backbone 
            feats = self.backbone(feats)

        if self.encoder is not None:
            encoding_phase_outputs = self.encoder(feats, Embeddings=self.decoder.get_embeddings())
            assert 'encoder_hidden_states' in encoding_phase_outputs.keys()
        else:
            encoding_phase_outputs = {'encoder_hidden_states': feats}

        inputs_for_predictor = encoding_phase_outputs.pop('inputs_for_predictor', encoding_phase_outputs)
        inputs_for_decoder = encoding_phase_outputs.pop('inputs_for_decoder', encoding_phase_outputs)
        if self.predictor is not None:
            predictor_outputs = self.predictor(
                word_embeddings=self.decoder.get_word_embeddings(),
                semantic_logits=semantic_logits,
                text_embs=text_embs,
                **inputs_for_predictor, 
                **kwargs,
            )
            inputs_for_decoder.update(predictor_outputs)

            if 'concat' in self.opt.get('use_attr_type', ''):
                inputs_for_decoder['encoder_hidden_states'] = torch.cat((inputs_for_decoder['encoder_hidden_states'], inputs_for_decoder['semantic_embs']), dim=1)

        return inputs_for_decoder
    
    def prepare_inputs_for_decoder(self, 
            encoding_phase_outputs: List[torch.Tensor], 
            batch: Dict[str, Any]
        ) -> Dict[str, torch.Tensor]:
        
        inputs_for_decoder = {}
        for key in self.input_keys_for_decoder:
            if key not in encoding_phase_outputs.keys() and \
                key not in batch.keys():
                raise KeyError('the input key `{}` can not be found in `encoding_phase_outputs` {} \
                    nor `batch` {}'.format(key, encoding_phase_outputs.keys(), batch.keys()))
            
            pointer = batch if key not in encoding_phase_outputs.keys() else encoding_phase_outputs
            inputs_for_decoder[key] = pointer[key]

        return inputs_for_decoder

    def decoding_phase(self,
            input_ids: torch.LongTensor,
            inputs_for_decoder: List[torch.Tensor], 
            last_time_step_logits: bool = False,
            **kwargs
        ) -> Dict[str, torch.Tensor]:
        
        raise NotImplementedError('Please implement this function in `TransformerSeq2Seq` or `RNNSeq2Seq`')
    
    def feedforward_step(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # the caption is already known, c.f. batch['input_ids']
        encoding_phase_outputs = self.encoding_phase(batch['feats'], **kwargs)
        inputs_for_decoder = self.prepare_inputs_for_decoder(encoding_phase_outputs, batch)
        
        # for RNN based decoder
        schedule_sampling_prob = 0
        if self.training and self.opt.get('scheduled_sampling_start', -1) >= 0:
            current_epoch = kwargs.get('current_epoch', None)
            assert current_epoch is not None, 'please pass the arguement `current_epoch`'

            if current_epoch > self.opt['scheduled_sampling_start']:
                frac = (current_epoch - self.opt['scheduled_sampling_start']) // self.opt['scheduled_sampling_increase_every']
                schedule_sampling_prob = min(self.opt['scheduled_sampling_increase_prob']  * frac, self.opt['scheduled_sampling_max_prob'])
                

        decoding_phase_outputs = self.decoding_phase(batch['input_ids'], inputs_for_decoder, 
                                                    schedule_sampling_prob=schedule_sampling_prob, **kwargs)
        
        return {**encoding_phase_outputs, **decoding_phase_outputs, 'schedule_sampling_prob': schedule_sampling_prob}
    
    def forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return self.feedforward_step(batch, **kwargs)
    
    
class TransformerSeq2Seq(Seq2SeqBase):
    def decoding_phase(self,
            input_ids: torch.LongTensor,
            inputs_for_decoder: List[torch.Tensor], 
            last_time_step_logits: bool = False,
            **kwargs
        ) -> Dict[str, torch.Tensor]:
            
        decoding_phase_outputs = self.decoder(
            input_ids, 
            **inputs_for_decoder, 
            **{**kwargs, 'last_time_step_logits': last_time_step_logits}
        )
        hidden_states = decoding_phase_outputs['hidden_states']

        if last_time_step_logits:
            logits = self.cls_head(hidden_states[:, -1, :])
        else:
            if not isinstance(hidden_states, list):
                logits = self.cls_head(hidden_states)
            else:
                logits = [self.cls_head(item) for item in hidden_states]
        
        decoding_phase_outputs['logits'] = logits

        pointer_outputs = {}
        if self.pointer is not None:
            pointer_outputs = self.pointer(**inputs_for_decoder, **decoding_phase_outputs, last_time_step_logits=last_time_step_logits)

        return {**decoding_phase_outputs, **pointer_outputs}


class RNNSeq2Seq(Seq2SeqBase):
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

    def decoding_phase(self,
            input_ids: torch.Tensor,
            inputs_for_decoder: List[torch.Tensor],
            decoder_rnn_hidden_states: Optional[torch.Tensor] = None,
            last_time_step_logits: bool = False,
            **kwargs
        ) -> Dict[str, torch.Tensor]:

        if last_time_step_logits:
            it = input_ids[:, -1] if input_ids.dim() == 2 else input_ids
            decoding_phase_outputs = self.decoder.forward_step(it=it, decoder_rnn_hidden_states=decoder_rnn_hidden_states, **inputs_for_decoder)
            decoding_phase_outputs['logits'] = self.cls_head(decoding_phase_outputs['hidden_states'])
        else:
            kwargs['cls_head'] = self.cls_head
            decoding_phase_outputs = self.decoder(input_ids, **inputs_for_decoder, **kwargs)
        
        return decoding_phase_outputs

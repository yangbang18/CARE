''' This module will handle the text generation with beam search. '''
import torch
import pickle
import numpy as np

from config import Constants
from misc.utils import auto_enlarge, get_shape_and_device
from misc.Decoding.Beam import Beam
from misc.Decoding.na_algorithms import MaskPredict, Left2Right, EasyFirst

__all__ = ('Translator_ARFormer', 'Translator_NARFormer')


def get_translator(opt: dict) -> object:
    class_name = 'Translator_{}'.format(opt['decoding_type'])
    if class_name not in globals():
        raise ValueError('We can not find the class `{}` in {}'.format(class_name, __file__))

    return globals()[class_name](opt)


class Translator_ARFormer(object):
    ''' 
        Load with trained model(s) and handle the beam search. 
        Note that model ensembling is available.
    '''
    def __init__(self, opt: dict = {}):
        super().__init__()
        self.beam_size = opt.get('beam_size', 5)
        self.beam_alpha = opt.get('beam_alpha', 1.0)
        self.topk = opt.get('topk', 1)
        self.max_len = opt.get('max_len', 30)
        self.ar_token_id = opt.get('ar_token_id', None)

    def translate_batch(self, models, batch, *args, **kwargs):
        with torch.no_grad():
            all_inputs_for_decoder = []
            all_decoder_rnn_hidden_states = [None] * len(models) # needed for RNN based decoders
            for index, model in enumerate(models):
                # handle model ensembling
                (bsz, *_), _ = get_shape_and_device(batch['feats'])
                
                # batch['feats'] is usually a list of torch.Tensor
                # but models.Wrapper.ModelEnsemble may feed a list of original batch['feats']
                if isinstance(batch['feats'][0], list):
                    encoding_phase_outputs = model.encoding_phase(batch['feats'][index])
                else:
                    encoding_phase_outputs = model.encoding_phase(batch['feats'])
                    
                inputs_for_decoder = model.prepare_inputs_for_decoder(encoding_phase_outputs, batch)
                inputs_for_decoder = auto_enlarge(inputs_for_decoder, self.beam_size, bsz) # repeat data for beam search
                all_inputs_for_decoder.append(inputs_for_decoder)
            
            (n_inst, *_), self.device = get_shape_and_device(all_inputs_for_decoder[0]['encoder_hidden_states'])
            n_inst //= self.beam_size # because the `encoder_hidden_states` has been enlarged

            #-- Prepare beams
            # TODO: add a variable `candidate_size`?
            inst_dec_beams = [
                Beam(self.beam_size, self.max_len, device=self.device, specific_nums_of_sents=self.topk, 
                    begin_of_sentence_id=self.ar_token_id if self.ar_token_id is not None else Constants.BOS
                ) 
                for _ in range(n_inst)
            ]

            #-- Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = self.get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            #-- Decode
            for len_input_ids in range(1, self.max_len):

                active_inst_idx_list, all_decoder_rnn_hidden_states = self.beam_decode_step(
                    models, inst_dec_beams, len_input_ids, 
                    all_inputs_for_decoder, all_decoder_rnn_hidden_states, inst_idx_to_position_map)

                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>

                all_inputs_for_decoder, all_decoder_rnn_hidden_states, inst_idx_to_position_map = self.collect_active_info(
                    all_inputs_for_decoder, all_decoder_rnn_hidden_states, inst_idx_to_position_map, active_inst_idx_list, self.beam_size)
        
        batch_hyps, batch_scores = self.collect_hypothesis_and_scores(inst_dec_beams, self.topk, self.beam_alpha)

        return batch_hyps, batch_scores

    def get_inst_idx_to_tensor_position_map(self, inst_idx_list):
        ''' Indicate the position of an instance in a tensor. '''
        return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

    def beam_decode_step(self, models, inst_dec_beams, len_input_ids, 
            all_inputs_for_decoder, all_decoder_rnn_hidden_states, inst_idx_to_position_map):
        ''' Decode and update beam status, and then return active beam idx '''

        n_active_inst = len(inst_idx_to_position_map)

        # Prepare beam input ids
        input_ids = [b.get_current_state() for b in inst_dec_beams if not b.done]
        input_ids = torch.stack(input_ids).to(self.device)
        input_ids = input_ids.view(-1, len_input_ids)

        word_probs, all_decoder_rnn_hidden_states = self.predict_word(
            models, input_ids, all_inputs_for_decoder, all_decoder_rnn_hidden_states, n_active_inst)

        # Update the beam with predicted word prob information and collect incomplete instances
        active_inst_idx_list = self.collect_active_inst_idx_list(
            inst_dec_beams, word_probs, inst_idx_to_position_map)

        return active_inst_idx_list, all_decoder_rnn_hidden_states

    def predict_word(self, models, input_ids, all_inputs_for_decoder, all_decoder_rnn_hidden_states, n_active_inst):
        word_probs = []

        new_all_decoder_rnn_hidden_states = []
        for model, inputs_for_decoder, decoder_rnn_hidden_states in \
                zip(models, all_inputs_for_decoder, all_decoder_rnn_hidden_states):
            
            decoding_phase_outputs = model.decoding_phase(
                input_ids=input_ids, 
                inputs_for_decoder=inputs_for_decoder, 
                decoder_rnn_hidden_states=decoder_rnn_hidden_states,
                last_time_step_logits=True
            )
            if 'probs' in decoding_phase_outputs:
                word_probs.append(torch.log(decoding_phase_outputs['probs']))
            else:
                word_probs.append(torch.log_softmax(decoding_phase_outputs['logits'], dim=1))
            new_all_decoder_rnn_hidden_states.append(decoding_phase_outputs.get('decoder_rnn_hidden_states', None))
        
        # average equally
        word_probs = torch.stack(word_probs, dim=0).mean(0) 
        word_probs = word_probs.view(n_active_inst, self.beam_size, -1)
        return word_probs, new_all_decoder_rnn_hidden_states

    def collect_active_inst_idx_list(self, inst_beams, word_probs, inst_idx_to_position_map):
        ''' Update beams with predicted word probs and collect active (incomplete) beams. '''
        active_inst_idx_list = []
        for inst_idx, inst_position in inst_idx_to_position_map.items():
            is_inst_complete = inst_beams[inst_idx].advance(word_probs[inst_position])

            if not is_inst_complete:
                active_inst_idx_list += [inst_idx]
        return active_inst_idx_list

    def collect_active_info(self, all_inputs_for_decoder, all_decoder_rnn_hidden_states, inst_idx_to_position_map, active_inst_idx_list, beam_size):
        ''' Collect the info of active (incomplete) beams on which the decoder will run. '''
        n_prev_active_inst = len(inst_idx_to_position_map)
        active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
        active_inst_idx = torch.LongTensor(active_inst_idx)
        
        args = (active_inst_idx, n_prev_active_inst, beam_size)

        new_all_inputs = []
        for inputs_for_decoder in all_inputs_for_decoder:
            new_all_inputs.append(self.auto_collect_active_part(inputs_for_decoder, *args))

        new_all_states = []
        for decoder_rnn_hidden_states in all_decoder_rnn_hidden_states:
            new_all_states.append(self.auto_collect_active_part(decoder_rnn_hidden_states, *args))

        active_inst_idx_to_position_map = self.get_inst_idx_to_tensor_position_map(active_inst_idx_list)

        return new_all_inputs, new_all_states, active_inst_idx_to_position_map
    
    def auto_collect_active_part(self, beamed_tensor, *args):
        ''' Collect tensor parts associated to active beams. '''
        if beamed_tensor is None:
            # this occurs when `beamed_tensor` belongs to the `decoder_rnn_hidden_states` 
            # and the decoder is not based on RNNs
            return None

        if isinstance(beamed_tensor, dict):
            # inputs_for_decoder
            return {
                key: self.auto_collect_active_part(beamed_tensor[key], *args)
                for key in beamed_tensor.keys()
            }
        elif isinstance(beamed_tensor, list):
            if isinstance(beamed_tensor[0], tuple):
                # this occurs when the decoder is multi-layer LSTMs
                # and `beamed_tensor` belongs to the `decoder_rnn_hidden_states` 
                return [
                    tuple([self.collect_active_part(_, *args) for _ in item])
                    for item in beamed_tensor
                ]
            return [self.collect_active_part(item, *args) for item in beamed_tensor]
        else:
            if isinstance(beamed_tensor, tuple):
                # this occurs when the decoder is a one-layer LSTM 
                # and `beamed_tensor` belongs to the `decoder_rnn_hidden_states` 
                return tuple([self.collect_active_part(item, *args) for item in beamed_tensor])
            return self.collect_active_part(beamed_tensor, *args)

    def collect_active_part(self, beamed_tensor, curr_active_inst_idx, n_prev_active_inst, beam_size):
        ''' Collect tensor parts associated to active instances. '''
        bsz, *d_hs = beamed_tensor.size()
        device = beamed_tensor.device

        if bsz == 1 and n_prev_active_inst * beam_size != 1:
            return beamed_tensor

        n_curr_active_inst = len(curr_active_inst_idx)
        new_shape = (n_curr_active_inst * beam_size, *d_hs)

        beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
        beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx.to(device))
        beamed_tensor = beamed_tensor.view(*new_shape)

        return beamed_tensor
    
    def collect_hypothesis_and_scores(self, inst_dec_beams, n_best, beam_alpha=1.0):
        all_hyp, all_scores = [], []
        for inst_idx in range(len(inst_dec_beams)):
            scores, tk = inst_dec_beams[inst_idx].sort_finished(beam_alpha)
            n_best = min(n_best, len(scores))
            all_scores += [scores[:n_best]]
            hyps = [inst_dec_beams[inst_idx].get_hypothesis_from_tk(t, k) for t, k in tk[:n_best]]
            all_hyp += [hyps]

        return all_hyp, all_scores


class Translator_NARFormer(object):
    def __init__(self, opt: dict = {}):
        super().__init__()
        self.opt = opt
        self.algorithms_mapping = {
            'mp': MaskPredict,
            'l2r': Left2Right,
            'ef': EasyFirst
        }
        self.paradigm = opt.get('paradigm', 'mp')
        assert self.paradigm in ['mp', 'l2r', 'ef']

        self.max_len = opt['max_len']
        self.length_beam_size = opt['length_beam_size']
        self.beam_alpha = opt.get('beam_alpha', 1.0)
        self.length_bias = opt.get('length_bias', 0)
    
    def translate_batch(self, models, batch, teacher_model_wrapper=None, vocab=None):
        if isinstance(models, list):
            assert len(models) == 1, 'only support the evaluation of single model now'
        
        model = models[0]

        (bsz, *_), device = get_shape_and_device(batch['feats'])
        encoding_phase_outputs = model.encoding_phase(batch['feats'])
        inputs_for_decoder = model.prepare_inputs_for_decoder(encoding_phase_outputs, batch)
        inputs_for_decoder = auto_enlarge(inputs_for_decoder, self.length_beam_size)
        
        if teacher_model_wrapper is not None:
            if not hasattr(self, 'vocab_mapping'):
                self.vocab_mapping = get_vocab_mapping(self.opt, teacher_model_wrapper.get_opt())
            
            teacher_model = teacher_model_wrapper.captioner
            teacher_model = teacher_model.to(device)
            teacher_model.eval()

            teacher_encoding_phase_outputs = teacher_model.encoding_phase(batch['feats'])
            teacher_inputs_for_decoder = teacher_model.prepare_inputs_for_decoder(teacher_encoding_phase_outputs, batch)
            teacher_inputs_for_decoder = auto_enlarge(teacher_inputs_for_decoder, self.length_beam_size)
        else:
            self.vocab_mapping = None
            teacher_model = None
            teacher_inputs_for_decoder = None
        
        # prepare tgt_tokens
        beam = self.predict_length_beam(encoding_phase_outputs, self.length_beam_size, self.length_bias)   
        self.length_beam_size = beam.shape[1]
        max_len = beam.max().item()

        add_eos = False
        tmp_tensor = batch['feats'][0]

        length_mask = torch.triu(tmp_tensor.new(max_len, max_len).fill_(1).long(), 1)
        length_mask = torch.stack([length_mask[beam[batch] - 1] for batch in range(bsz)], dim=0)

        tgt_tokens = tmp_tensor.new(bsz, self.length_beam_size, max_len).fill_(Constants.MASK).long()
        tgt_tokens = (1 - length_mask) * tgt_tokens + length_mask * Constants.PAD
        tgt_tokens = tgt_tokens.view(bsz * self.length_beam_size, max_len)

        if add_eos:
            for i, length in enumerate(beam.view(bsz * self.length_beam_size)):
                tgt_tokens[i, length] = Constants.EOS
        
        # run the algorithm
        algorithm = self.algorithms_mapping[self.paradigm](self.opt, self.vocab_mapping, vocab)
        hypotheses, lprobs = algorithm.generate(
            model, teacher_model, inputs_for_decoder, teacher_inputs_for_decoder, tgt_tokens)
        
        # get the best performed hypotheses and lprobs
        hypotheses = hypotheses.view(bsz, self.length_beam_size, max_len)
        lprobs = lprobs.view(bsz, self.length_beam_size, max_len)
        
        tgt_lengths = (1 - length_mask).sum(-1)
        tgt_lengths = tgt_lengths.view(bsz, self.length_beam_size)
        avg_log_prob = lprobs.sum(-1) / (tgt_lengths.float() ** self.beam_alpha)
        best_lengths = avg_log_prob.max(-1)[1]                                          # [batch_size]
        best_lengths = best_lengths.unsqueeze(1).unsqueeze(2).repeat(1, 1, max_len)     # [batch_size, 1, max_len]
        
        
        hypotheses = hypotheses.gather(1, best_lengths) # [batch_size, 1, max_len]
        lprobs = lprobs.gather(1, best_lengths) # [batch_size, 1, max_len]
        
        return hypotheses.cpu().tolist(), lprobs.cpu().tolist()

    def predict_length_beam(self, encoding_phase_outputs, length_beam_size, length_bias):
        if 'preds_length' in encoding_phase_outputs:
            beam = encoding_phase_outputs['preds_length'].topk(length_beam_size, dim=1)[1] + length_bias
            beam[beam < 4] = 4
            beam[beam > self.max_len] = self.max_len
        else:
            length_range = self.opt.get('na_length_range', [5, 11])
            (bsz, *_), device = get_shape_and_device(encoding_phase_outputs['encoder_hidden_states'])
            beam = torch.arange(*length_range, device=device, dtype=torch.long)
            beam = beam.unsqueeze(0).repeat(bsz, 1)

        return beam 


def get_vocab_mapping(opt, teacher_opt):
    if teacher_opt is None:
        return None

    vocab = pickle.load(open(opt["info_corpus"], 'rb'))['info']['itow']
    teacher_vocab = pickle.load(open(teacher_opt["info_corpus"], 'rb'))['info']['itow']
    if vocab == teacher_vocab:
        return None

    teacher_w2ix = {v: k for k, v in teacher_vocab.items()}
    
    vocab_mapping = np.zeros((len(vocab), ))
    for k, v in vocab.items():
        vocab_mapping[int(k)] = int(teacher_w2ix[v])
    
    vocab_mapping = torch.from_numpy(vocab_mapping).long()
    
    assert vocab_mapping[Constants.PAD] == Constants.PAD
    return vocab_mapping

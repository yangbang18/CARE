from config import Constants
import torch
from tqdm import tqdm
import torch.nn.functional as F

def generate_step_with_prob(out, zeros=[]):
    probs = F.softmax(out, dim=-1)
    for item in zeros:
        if len(probs.shape) == 3:
            probs[:, :, item] = 0
        else:
            probs[:, item] = 0
    max_probs, idx = probs.max(dim=-1)
    return idx, max_probs, probs

def to_sentence_with_prob(hyp, prob, vocab, break_words=[Constants.PAD], skip_words=[]):
    tokens = []
    for word_id, p in zip(hyp, prob):
        if word_id in skip_words:
            continue
        if word_id in break_words:
            break
        tokens.append('%12s(%.2f)'%(vocab[word_id], p))
    return ' '.join(tokens)

class Algorithm_Base(object):
    """docstring for Algorithm_Base"""
    def __init__(self, opt, vocab_mapping, vocab):
        super(Algorithm_Base, self).__init__()
        # knowledge distillation
        self.vocab_mapping = vocab_mapping

        # if masking_decision = True, teacher will be used to rescore the intermediate sequences
        # if no_candidate_decision = False, teacher will be used to rescore the final sequences
        self.masking_decision = opt.get('masking_decision', False)
        self.no_candidate_decision = opt.get('no_candidate_decision', False)

        self.visual_tag = Constants.VIS

        # use to print sentences if we want
        self.vocab = vocab # itow
        self.wtoi = {w: i for i, w in self.vocab.items()}
        self.algorithm_print_sent = opt.get('algorithm_print_sent', False)

        self.opt = opt

    def prepare(self, tgt_tokens):
        bsz, seq_len = tgt_tokens.size()

        self.eos_mask = tgt_tokens.eq(Constants.EOS)
        self.special_token_mask = self.eos_mask

        pad_mask = tgt_tokens.eq(Constants.PAD)
        seq_lens = seq_len - pad_mask.sum(dim=1)
        return tgt_tokens, pad_mask, seq_lens
    
    def postprocess(self, tgt_tokens, lprobs):
        return tgt_tokens, lprobs

    def get_coarse_grained_templates(self, model, inputs_for_decoder, tgt_tokens, pad_mask, **kwargs):
        mask_ind = tgt_tokens.eq(Constants.MASK)
        tgt_tokens[mask_ind] = self.visual_tag
        tgt_tokens, token_probs = self.generate_non_autoregressive(model, inputs_for_decoder, tgt_tokens, pad_mask, **kwargs)
        token_probs[tgt_tokens.eq(Constants.MASK)] = 0.0
        return tgt_tokens, token_probs

    def generate_non_autoregressive(self, model, inputs_for_decoder, tgt_tokens, pad_mask, zeros=[], **kwargs):
        decoding_phase_outputs = model.decoding_phase(
            input_ids=tgt_tokens, inputs_for_decoder=inputs_for_decoder)

        tgt_tokens, token_probs, all_probs = generate_step_with_prob(
            decoding_phase_outputs['logits'], zeros=zeros)

        tgt_tokens[pad_mask] = Constants.PAD
        token_probs[pad_mask] = 1.0

        if hasattr(self, 'special_token_mask'):
            token_probs[self.special_token_mask] = 1.0

        tgt_tokens[self.eos_mask] = Constants.EOS

        return tgt_tokens, token_probs

    def mapping(self, tgt_tokens):
        if self.vocab_mapping.device != tgt_tokens.device:
            self.vocab_mapping = self.vocab_mapping.to(tgt_tokens.device)

        tokens = tgt_tokens.clone().flatten()
        tokens = self.vocab_mapping[tgt_tokens]
        return tokens.view(*tgt_tokens.shape)

    def scoring_by_teacher(self, teacher_model, teacher_inputs_for_decoder, tgt_tokens, pad_mask, is_last=False, **kwargs):
        all_ones = tgt_tokens.new(*tgt_tokens.shape).fill_(1).float()
        
        if teacher_model is None:
            return all_ones

        if is_last:
            if self.no_candidate_decision:
                return all_ones
        else:
            if not self.masking_decision:
                return all_ones

        # if we use knowledge distillation, we should map the tokens
        tokens = self.mapping(tgt_tokens) if self.vocab_mapping is not None else tgt_tokens

        # add the <bos> token to the start of the sequences
        tgt_tokens_with_bos = torch.cat([tokens.new(tokens.size(0), 1).fill_(Constants.BOS), tokens], dim=1)

        # forward
        teacher_decoding_phase_outputs = teacher_model.decoding_phase(
            input_ids=tgt_tokens_with_bos[:, :-1], inputs_for_decoder=teacher_inputs_for_decoder)

        probs = F.softmax(teacher_decoding_phase_outputs['logits'], dim=-1)
        
        # get the possibility of p(y_t | y_<t, R)
        probs = probs.gather(2, tokens.unsqueeze(2)).squeeze(2)
        
        # mask sure the possibility of <pad> tokens is 1.0
        probs[pad_mask] = 1.0

        if not is_last:
            probs[self.eos_mask] = 1.0

        return probs

    def select_worst(self, token_probs, num_mask):
        """
            for each example i
            select num_mask[i] tokens that the model is least confident about to mask out
        """
        masks = torch.zeros(*token_probs.shape, device=token_probs.device)
        for i in range(masks.size(0)):
            ind = token_probs[i, :].topk(max(1, num_mask[i]), largest=False, sorted=False)[1]
            masks[i, ind] = 1
        return masks.bool()

    def print_sent(self, tgt_tokens, token_probs, counter, debug=False):
        if self.algorithm_print_sent or debug:
            sample_ind = 0
            tqdm.write("Iteration %2d: "%counter + \
                to_sentence_with_prob(tgt_tokens[sample_ind].tolist(), token_probs[sample_ind].tolist(), self.vocab)) 


class MaskPredict(Algorithm_Base):
    def __init__(self, opt, vocab_mapping, vocab):
        super().__init__(opt, vocab_mapping, vocab)
        self.use_ct = opt.get('use_ct', False)
        self.T = opt.get('iterations', 5)

    def generate(self, model, teacher_model, inputs_for_decoder, teacher_inputs_for_decoder, tgt_tokens):
        tgt_tokens, pad_mask, seq_lens = self.prepare(tgt_tokens)
        kwargs = dict(
            model=model,
            teacher_model=teacher_model,
            inputs_for_decoder=inputs_for_decoder,
            teacher_inputs_for_decoder=teacher_inputs_for_decoder,
            pad_mask=pad_mask
        )

        if self.use_ct:
            tgt_tokens, token_probs = self.get_coarse_grained_templates(tgt_tokens=tgt_tokens, **kwargs)
        else:
            tgt_tokens, token_probs = self.generate_non_autoregressive(tgt_tokens=tgt_tokens, **kwargs)
        
        # if we use coarse-grained templates, it will take one more iteration
        T = self.T + 1 if self.use_ct else self.T
              
        self.print_sent(tgt_tokens, token_probs, counter=0)

        for counter in range(1, T):
            corresponding_probs = self.scoring_by_teacher(tgt_tokens=tgt_tokens, is_last=False, **kwargs)

            if self.use_ct and counter == 1:
                # if we use coarse-grained templates, we first complete the sequences
                # i.e., sentence making in Fig. 1(b) in the paper
                mask_ind = (tgt_tokens == Constants.MASK)
            else:
                ratio = (1.0 - (counter / T))
                num_mask = (seq_lens.float() * ratio).long()
                mask_ind = self.select_worst(token_probs * corresponding_probs, num_mask)

            # Mask
            tgt_tokens[mask_ind] = Constants.MASK
            # Predict
            new_tgt_tokens, new_token_probs = self.generate_non_autoregressive(tgt_tokens=tgt_tokens, **kwargs)
            # only update those masked tokens and their possibilities
            tgt_tokens[mask_ind] = new_tgt_tokens[mask_ind]
            token_probs[mask_ind] = new_token_probs[mask_ind]

            self.print_sent(tgt_tokens, token_probs, counter=counter)

        # teacher rescoring
        corresponding_probs = self.scoring_by_teacher(tgt_tokens=tgt_tokens, is_last=True, **kwargs)
        lprobs = (token_probs * corresponding_probs).log()
        return self.postprocess(tgt_tokens, lprobs)


class Left2Right(Algorithm_Base):
    def __init__(self, opt, vocab_mapping, vocab):
        super().__init__(opt, vocab_mapping, vocab)
        self.use_ct = opt.get('use_ct', False)
        self.T = opt.get('q_iterations', 1)
        self.q = opt.get('q', 1)

    def generate(self, model, teacher_model, inputs_for_decoder, teacher_inputs_for_decoder, tgt_tokens):
        bsz, seq_len = tgt_tokens.size()
        pad_mask = tgt_tokens.eq(Constants.PAD)
        seq_lens = seq_len - pad_mask.sum(dim=1)
        
        if self.use_ct:
            tgt_tokens, token_probs = self.get_coarse_grained_templates(model, inputs_for_decoder, tgt_tokens, pad_mask)
            visual_mask = tgt_tokens.ne(Constants.MASK) & tgt_tokens.ne(Constants.PAD)
        else:
            token_probs = tgt_tokens.new(*tgt_tokens.shape).fill_(0).float()
            token_probs[pad_mask] = 1.0

        def get_mask_ind(tgt_tokens, seq_lens):
            all_mask_ind = []
            for i in range(tgt_tokens.size(0)):
                item = [j for j in range(seq_lens[i]) if tgt_tokens[i, j] == Constants.MASK]
                all_mask_ind.append(item)
            return all_mask_ind

        def select_left(all_mask_ind, current, q):
            masks = torch.zeros(*token_probs.shape, device=token_probs.device)
            for i in range(masks.size(0)):
                ind = all_mask_ind[i][current:min(current+q,len(all_mask_ind[i]))] if current < len(all_mask_ind[i]) else []
                masks[i, ind] = 1
            return masks.bool()

        all_mask_ind = get_mask_ind(tgt_tokens, seq_lens)

        for counter in range(0, seq_len, self.q):
            mask_ind = select_left(all_mask_ind, counter, self.q)
            if mask_ind.sum() == 0: break

            tgt_tokens[mask_ind] = Constants.MASK
            # Predict
            new_tgt_tokens, new_token_probs = self.generate_non_autoregressive(model, inputs_for_decoder, tgt_tokens, pad_mask)
            
            token_probs[mask_ind] = new_token_probs[mask_ind]
            tgt_tokens[mask_ind] = new_tgt_tokens[mask_ind]

        for i in range(self.T):
            if i == 0 and self.use_ct:
                mask_ind = visual_mask
            else:
                refine_ratio = 0.4 * (1.0 - (i / self.T))
                num_mask = (seq_lens.float() * refine_ratio).long()
                mask_ind = self.select_worst(token_probs, num_mask)

            tgt_tokens[mask_ind] = Constants.MASK
            new_tgt_tokens, new_token_probs = self.generate_non_autoregressive(model, inputs_for_decoder, tgt_tokens, pad_mask)

            token_probs[mask_ind] = new_token_probs[mask_ind]
            tgt_tokens[mask_ind] = new_tgt_tokens[mask_ind]

        corresponding_probs = self.scoring_by_teacher(teacher_model, teacher_inputs_for_decoder, tgt_tokens, pad_mask, is_last=True)
        lprobs = (token_probs * corresponding_probs).log()

        return tgt_tokens, lprobs


class EasyFirst(Algorithm_Base):
    def __init__(self, opt, vocab_mapping, vocab):
        super().__init__(opt, vocab_mapping, vocab)
        self.use_ct = opt.get('use_ct', False)
        self.T = opt.get('q_iterations', 1)
        self.q = opt.get('q', 1)

    def generate(self, model, teacher_model, inputs_for_decoder, teacher_inputs_for_decoder, tgt_tokens):
        bsz, seq_len = tgt_tokens.size()
        pad_mask = tgt_tokens.eq(Constants.PAD)
        seq_lens = seq_len - pad_mask.sum(dim=1)
        
        if self.use_ct:
            tgt_tokens, token_probs = self.get_coarse_grained_templates(model, inputs_for_decoder, tgt_tokens, pad_mask)
            visual_mask = tgt_tokens.ne(Constants.MASK) & tgt_tokens.ne(Constants.PAD)
        else:
            token_probs = tgt_tokens.new(*tgt_tokens.shape).fill_(0).float()
            token_probs[pad_mask] = 1.0

        def select_most_confidence(token_probs, mask_ind, q):
            masks = torch.zeros(*token_probs.shape, device=token_probs.device)
            token_probs[~mask_ind] = 0
            remain_length = mask_ind.sum(-1)
            for i in range(masks.size(0)):
                if remain_length[i] == 0:
                    ind = []
                else:
                    ind = token_probs[i, :].topk(min(q, remain_length[i]), largest=True, sorted=False)[1]
                masks[i, ind] = 1
            return masks.bool()

        counter, pre = 0, 0
        while True:
            counter += 1
            mask_ind = tgt_tokens.eq(Constants.MASK)

            remain = mask_ind.sum()
            if remain == 0 or pre == remain: # to avoid dead loop
                break
            pre = remain

            new_tgt_tokens, new_token_probs = self.generate_non_autoregressive(model, inputs_for_decoder, tgt_tokens, pad_mask)

            most_confidence_ind = select_most_confidence(new_token_probs, mask_ind, self.q)
            token_probs[most_confidence_ind] = new_token_probs[most_confidence_ind]
            tgt_tokens[most_confidence_ind] = new_tgt_tokens[most_confidence_ind]
        
        for i in range(self.T):
            if i == 0 and self.use_ct:
                mask_ind = visual_mask
            else:
                refine_ratio = 0.4 * (1.0 - (i / self.T))
                num_mask = (seq_lens.float() * refine_ratio).long()
                mask_ind = self.select_worst(token_probs, num_mask)

            tgt_tokens[mask_ind] = Constants.MASK
            new_tgt_tokens, new_token_probs = self.generate_non_autoregressive(model, inputs_for_decoder, tgt_tokens, pad_mask)
            token_probs[mask_ind] = new_token_probs[mask_ind]
            tgt_tokens[mask_ind] = new_tgt_tokens[mask_ind]
        
        corresponding_probs = self.scoring_by_teacher(teacher_model, teacher_inputs_for_decoder, tgt_tokens, pad_mask, is_last=True)
        lprobs = (token_probs * corresponding_probs).log()

        return tgt_tokens, lprobs

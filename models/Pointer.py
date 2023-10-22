import torch
import torch.nn as nn
from config import Constants
from models.components.Attention import ScaledDotProductAttention


def get_pointer(opt: dict) -> nn.Module:
    class_name = opt.get('pointer', None)
    if class_name is None:
        return None

    if class_name not in globals():
        raise ValueError('We can not find the class `{}` in {}'.format(class_name, __file__))

    return globals()[class_name](opt)


class Pointer(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.attention = ScaledDotProductAttention(
            dim_hidden=opt['dim_hidden'],
            num_attention_heads=1,
            attention_probs_dropout_prob=opt['attention_probs_dropout_prob'],
        )
        self.Wq = nn.Linear(opt['dim_hidden'], 1, bias=True)
        self.Wc = nn.Linear(opt['dim_hidden'], 1, bias=True)
        self.act = nn.Sigmoid()
        self.copy_scale = opt.get('copy_scale', 1)
    
    def forward(self, hidden_states, ret_text_embs, ret_input_ids, logits, last_time_step_logits=False, **kwargs):
        if last_time_step_logits:
            hidden_states = hidden_states[:, -1:, :]

        assert ret_input_ids.dim() == 3, f'expected shape: (bsz, n_retrieval, seq_len); obtained shape {ret_input_ids.shape}'
        assert ret_text_embs.dim() == 4, f'expected shape: (bsz, n_retrieval, seq_len, dim_hidden); obtained shape {ret_text_embs.shape}'
        assert ret_input_ids.shape == ret_text_embs.shape[:3]

        bsz, n_retrieval, seq_len = ret_input_ids.shape
        T = hidden_states.size(1)

        # (bsz * n_retrieval, T, dim_hidden)
        hidden_states = hidden_states.unsqueeze(1).repeat(1, n_retrieval, 1, 1).view(bsz * n_retrieval, *hidden_states.shape[1:])
        ret_text_embs = ret_text_embs.view(bsz * n_retrieval, seq_len, -1) # (bsz * n_retrieval, seq_len, dim_hidden)

        ret_input_ids = ret_input_ids.view(bsz * n_retrieval, 1, seq_len)
        ret_attention_mask = ret_input_ids.eq(Constants.PAD)

        # context: (bsz * n_retrieval, T, dim_hidden)
        # attention_probs: (bsz * n_retrieval, n_heads, T, seq_len)
        context, attention_probs = self.attention(
            q=hidden_states,
            k=ret_text_embs,
            v=ret_text_embs,
            attention_mask=ret_attention_mask,
        )

        p_copy = self.copy_scale * self.act(self.Wq(hidden_states) + self.Wc(context)).view(bsz, n_retrieval, T, 1)

        ret_input_ids = ret_input_ids.repeat(1, T, 1).view(bsz, n_retrieval, T, seq_len)
        ret_probs = p_copy * attention_probs.mean(1).view(bsz, n_retrieval, T, seq_len)

        if last_time_step_logits:
            logits = logits[:, None, None, :]
        else:
            logits = logits[:, None, :, :]
          
        probs = (1 - p_copy) * logits.softmax(dim=-1).repeat(1, n_retrieval, 1 ,1) # (bsz, n_retrieval, T, vocab_size)
        probs.scatter_add_(dim=3, index=ret_input_ids, src=ret_probs)
        probs = probs.mean(1).squeeze(1)

        #print(p_copy.mean().detach().cpu().item())
        return {'probs': probs}


import os
import torch
import torch.nn as nn
import math
import numpy as np
from typing import Optional, Union
from config import Constants


def get_prj_by_flag(opt, prj: Union[nn.ModuleList, nn.Module], flag: Optional[str]=None) -> nn.Module:
    if isinstance(prj, nn.ModuleList):
        assert flag is not None
        return prj[opt['attribute_prediction_flags'].index(flag)]
    return prj


def prepare_merged_probs(
        scores: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        return_avg_prob: bool = False
    ):
    assert len(scores.shape) == 3, "[bsz, seq_len, n_attributes]"
    probs = torch.sigmoid(scores)
    raw = torch.log(torch.clamp(1.0 - probs, 1e-12, 1)) # avoid nan

    if mask is not None:
        # if a position's mask is True, then its prob is 0, so log(1 - prob) is 0 too
        mask = mask.to(scores.device)
        raw = raw.masked_fill(mask.unsqueeze(2).expand_as(raw), 0)

        denominator = (~mask).sum(dim=1).float()
        denominator = torch.where(
            denominator > 0, 
            denominator, 
            torch.ones_like(denominator).to(denominator.device)
        ) # avoid nan
        
        avg_prob = probs.mean(dim=2) # (bsz, seq_len)
        avg_prob = torch.sum(avg_prob * (~mask).float(), dim=1) / denominator # (bsz, )
    else:
        avg_prob = probs.mean(dim=(1, 2)) # (bsz, )

    merge = raw.sum(dim=1)
    outputs =  1.0 - torch.exp(merge)

    return (outputs, avg_prob) if return_avg_prob else outputs


class Predictor_attribute(nn.Module):
    def __init__(self, opt):
        super(Predictor_attribute, self).__init__()
        self.opt = opt 
        self.flags = opt['attribute_prediction_flags']
        self.sparse_sampling = opt.get('attribute_prediction_sparse_sampling', False)

        self.channel_concat = opt.get('attribute_prediction_channel_concat', False)
        self.mean_pooling = opt.get('attribute_prediction_mean_pooling', False)

        self.modality = opt.get('modality_for_predictor', None) or opt['modality']

        if opt.get('attribute_prediction_share_prj', False) or len(self.flags) == 1:
            self.prj = nn.Linear(
                opt['dim_hidden'] * (len(self.modality) if self.channel_concat else 1), 
                opt['attribute_prediction_k']
            )
        else:
            self.prj = nn.ModuleList([
                nn.Linear(opt['dim_hidden'], opt['attribute_prediction_k']) 
                for _ in range(len(self.flags))
            ])
    
    def get_topk_attribute_predictions(self, feats, mask=None, topk=100, flag=None):
        scores = get_prj_by_flag(self.opt, self.prj, flag)(feats)
        preds_attr = prepare_merged_probs(scores, mask=mask, return_avg_prob=False) # [bsz, n_attributes]
        topk_probs, topk_indices = preds_attr.topk(topk, dim=-1, largest=True, sorted=True)
        return topk_probs, topk_indices

    def forward(self, encoder_hidden_states, **kwargs):
        '''
            encoder_hidden_states: [bsz, n_frames * n_modality, d] when `fusion` == 'temporal_concat'
                                   list of [bsz, n_frames, d] when `fusion` == 'none'
        '''
        if isinstance(encoder_hidden_states, list):
            hidden_states = torch.cat(encoder_hidden_states, dim=1) # (bsz, n_frames * n_modality, dim_hidden)
        else:
            hidden_states = encoder_hidden_states
        
        if self.channel_concat and self.mean_pooling:
            hidden_states = torch.cat(kwargs['mean_encoder_hidden_states'], dim=-1).unsqueeze(1) # (bsz, 1, n_modality * dim_hidden)
        elif self.channel_concat:
            n_modality = len(self.modality)
            assert hidden_states.size(1) % n_modality == 0
            hidden_states = hidden_states.chunk(chunks=n_modality, dim=1)
            hidden_states = torch.cat(hidden_states, dim=-1) # (bsz, n_frames, n_modality * dim_hidden)
            assert hidden_states.size(1) == self.opt['n_frames']
        elif self.mean_pooling:
            hidden_states = torch.stack(kwargs['mean_encoder_hidden_states'], dim=1) # (bsz, n_modality, dim_hidden)

        mask = None
        if self.training and self.sparse_sampling:
            assert 'V' in self.flags
            bsz, seq_len = hidden_states.shape[:2]
            all_ids = [_ for _ in range(seq_len)]
            sampled_hidden_states = []

            mask = hidden_states.new(bsz, seq_len).fill_(1)
            for i in range(bsz):
                sparsely_sampling_ratio = np.random.rand()
                sparsely_sampling_num = math.ceil(seq_len * sparsely_sampling_ratio)

                sampled_ids = np.random.choice(all_ids, sparsely_sampling_num, replace=False)
                sampled_ids = list(sampled_ids) + [0] * (seq_len - sparsely_sampling_num) # padding
                
                sampled_hidden_states.append(hidden_states[i, sampled_ids])
                mask[i, :sparsely_sampling_num] = 0
                
            hidden_states = torch.stack(sampled_hidden_states, dim=0) # [bsz, seq_len, dim_hidden]
            mask = mask.bool()
            assert hidden_states.shape[1] == seq_len
        
        if 'V' not in self.flags:
            preds_attr, avg_prob_attr = None, None
        else:
            scores = get_prj_by_flag(self.opt, self.prj, flag='V')(hidden_states)
            preds_attr, avg_prob_attr = prepare_merged_probs(scores, return_avg_prob=True, mask=mask) # [bsz, n_attributes]

        return {
            'preds_attr': preds_attr, 
            'avg_prob_attr': avg_prob_attr, 
            'attribute_prediction_prj': self.prj,
        }

    @staticmethod
    def add_specific_args(parent_parser: object) -> object:
        parser = parent_parser.add_argument_group(title='Attribute Prediction Settings')
        parser.add_argument('-ap', '--attribute_prediction', default=False, action='store_true')
        parser.add_argument('-ap_k', '--attribute_prediction_k', type=int, default=500)

        parser.add_argument('-apcc', '--attribute_prediction_channel_concat', action='store_true')
        parser.add_argument('-apmp', '--attribute_prediction_mean_pooling', action='store_true')

        parser.add_argument('-ap_flags', '--attribute_prediction_flags', type=str, default='V')
        parser.add_argument('-ap_scales', '--attribute_prediction_scales', type=float, nargs='+', default=[1.0])

        parser.add_argument('-ap_ss', '--attribute_prediction_sparse_sampling', default=False, action='store_true')
        parser.add_argument('-ap_sp', '--attribute_prediction_share_prj', default=False, action='store_true')
        
        parser.add_argument('--TAP_pos', default=False, action='store_true')
        parser.add_argument('--TAP_ln', default=False, action='store_true')

        # new
        parser.add_argument('--retrieval', action='store_true')

        parser.add_argument('--retrieval_unique_max_len', type=int, default=50)
        parser.add_argument('-rtopk', '--retrieval_topk', type=int, default=20)
        parser.add_argument('--retrieval_arch', type=str, default='ViT')

        parser.add_argument('--modality_for_decoder', type=str)
        parser.add_argument('--modality_for_predictor', type=str)
        parser.add_argument('-dm_flags', '--decoder_modality_flags', type=str, choices=['V', 'VA', 'VAT', 'VT', 'I'])
        parser.add_argument('-pm_flags', '--predictor_modality_flags', type=str, choices=['V', 'VA', 'VAT', 'VT', 'A', 'T', 'IT'])

        parser.add_argument('-gsg_not_detach', '--global_semantic_guidance_not_detach', action='store_true')

        parser.add_argument('-ahab', '--add_hybrid_attention_bias', action='store_true')
        return parent_parser
    
    @staticmethod
    def check_args(args: object) -> None:
        if args.attribute_prediction:
            if not isinstance(args.crits, list):
                args.crits = [args.crits]
            if 'attribute' not in args.crits:
                args.crits.append('attribute')
        
        base_path = os.path.join(Constants.base_data_path, args.dataset, 'retrieval')
        arch_mapping = {
            'ViT': (512, os.path.join(base_path, 'CLIP_ViT-B-32_unique.hdf5')),
            'ViT16': (512, os.path.join(base_path, 'CLIP_ViT-B-16_unique.hdf5')),
            'RN101': (512, os.path.join(base_path, 'CLIP_RN101_unique.hdf5')),
            'RN50': (1024, os.path.join(base_path, 'CLIP_RN50_unique.hdf5')),
            'RN50x4': (640, os.path.join(base_path, 'CLIP_RN50x4_unique.hdf5')),
            'RN50x16': (768, os.path.join(base_path, 'CLIP_RN50x16_unique.hdf5')),
        }

        if args.retrieval:
            assert getattr(args, 'pointer', None) is not None
            args.modality = args.modality + 't'
            args.dim_t, args.feats_t = arch_mapping[args.retrieval_arch]
        
        if args.attribute_prediction:
            assert args.feats, "Please specify --feats"
            if not any(k in args.task for k in ['VAP', 'TAP', 'DAP']):
                assert args.decoder_modality_flags, "Please specify --decoder_modality_flags instead of --modality"
                assert args.predictor_modality_flags, "Please specify --predictor_modality_flags instead of --modality"

                args.modality_for_decoder = Constants.flag2modality[args.decoder_modality_flags]
                args.modality_for_predictor = Constants.flag2modality[args.predictor_modality_flags]

                _all = args.modality_for_decoder + args.modality_for_predictor
                args.modality = ''
                for char in ['a', 'm', 'i', 'r']:
                    if char in _all:
                        args.modality = args.modality + char
            
            if getattr(args, 'pointer', None):
                args.modality = args.modality + 't'
            
            if 'r' in args.modality:
                args.dim_r, args.feats_r = arch_mapping[args.retrieval_arch]


class TextPostProcesser(nn.Module):
    def __init__(self, opt):
        super(TextPostProcesser, self).__init__()
        if opt.get('TAP_pos', False):
            self.PE = nn.Embedding(opt['max_len'], opt['dim_hidden'])

        if opt.get('TAP_ln', False):
            self.LN = nn.LayerNorm(opt['dim_hidden'], eps=opt['layer_norm_eps'])
        
        self.dropout = nn.Dropout(opt['hidden_dropout_prob'])

    def forward(self, word_embeddings):
        if hasattr(self, 'PE'):
            seq_length = word_embeddings.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=word_embeddings.device)
            position_ids = position_ids.unsqueeze(0).repeat(word_embeddings.size(0), 1)
            position_embeddings = self.PE(position_ids)
            word_embeddings = word_embeddings + position_embeddings
        
        if hasattr(self, 'LN'):
            word_embeddings = self.LN(word_embeddings)
        
        word_embeddings = self.dropout(word_embeddings)
        return word_embeddings


class SemanticContainer(nn.Module):
    def __init__(self, opt):
        super(SemanticContainer, self).__init__()

        from models.components.Embeddings import NaiveEmbeddings
        self.attr_embs = NaiveEmbeddings(
            n_words=opt['attribute_prediction_k'],
            n_positions=opt['use_attr_topk'],
            dim_hidden=opt['dim_hidden'],
            layer_norm_eps=opt['layer_norm_eps'],
            hidden_dropout_prob=opt['hidden_dropout_prob'],
            padding_idx=None,
            has_dropout=not opt.get('attr_embs_no_dropout', False),
        ) if 'L0' not in opt.get('use_attr_flags', '') else None

        self.topk = opt['use_attr_topk']
        self.opt = opt

        self.latent_topic_flag = False
        if 'emb' in opt.get('use_attr_type', ''):
            self.latent_topic_flag = True
            self.semantic2hidden = nn.Linear(opt['attribute_prediction_k'], opt['dim_hidden'], bias='pp_emb' in opt.get('use_attr_type', ''))
        
    def forward(self, encoder_hidden_states, preds_attr=None, semantic_logits=None, **kwargs):
        if semantic_logits is None:
            _, semantic_labels = preds_attr.topk(self.topk, dim=1, sorted=True, largest=True)
        else:
            assert isinstance(semantic_logits, (list, tuple))
            semantic_logits = [prepare_merged_probs(logits) for logits in semantic_logits]
            semantic_labels = [logits.topk(self.topk, dim=1, sorted=True, largest=True)[1] for logits in semantic_logits]
            semantic_labels = torch.cat(semantic_labels, dim=-1)
            
        input_ids, input_embs = semantic_labels, None
        semantic_embs = None

        if self.attr_embs is not None:
            semantic_embs = self.attr_embs(input_ids, input_embs)
        
        semantic_hidden_states = None
        if self.latent_topic_flag:
            semantic_hidden_states = self.semantic2hidden(preds_attr if self.opt.get('global_semantic_guidance_not_detach') else preds_attr.detach())
        else:
            semantic_hidden_states = None

        outputs = {
            'semantic_embs': semantic_embs, 
            'semantic_labels': semantic_labels,
            'semantic_hidden_states': semantic_hidden_states
        }

        return outputs

    @staticmethod
    def add_specific_args(parent_parser: object) -> object:
        parser = parent_parser.add_argument_group(title='Semantic Container Settings')
        parser.add_argument('--use_attr', action='store_true')
        parser.add_argument('--use_attr_type', type=str, default='')
        parser.add_argument('--use_attr_flags', type=str, default='G1Lc')
        parser.add_argument('--use_attr_topk', type=int, default=30)
        
        parser.add_argument('--attr_layer_pos', type=str, default='cross2attr', choices=['cross2attr', 'attr2cross', 'parallel'])

        # semantic composition
        parser.add_argument('--compositional_intra', action='store_true')
        parser.add_argument('--compositional_inter', action='store_true')
        parser.add_argument('--compositional_ffn', action='store_true')
        parser.add_argument('--dim_factor_scale', type=int, default=2)
        return parent_parser
    
    @staticmethod
    def check_args(args: object) -> None:
        if not args.use_attr_type and args.use_attr_flags == 'G0L0':
            args.use_attr = False

        if args.use_attr:
            assert hasattr(args, 'attribute_prediction')
            assert args.attribute_prediction, '`attribute_prediction` should be True when you want to use predicted attributes in the model'

            if not args.use_attr_type:
                mapping = {
                    'G0': '',
                    'G1': 'emb',
                    'Gp': 'pp_emb',
                    'L0': '',
                    'L1': 'att',
                    'Lc': 'concat',
                }
                assert len(args.use_attr_flags) == 4
                args.use_attr_type = mapping[args.use_attr_flags[:2]] + '_' + mapping[args.use_attr_flags[2:]]

        def add_predictor(args):
            if not hasattr(args, 'predictors_to_be_added'):
                args.predictors_to_be_added = ['SemanticContainer']
            else:
                if not isinstance(args.predictors_to_be_added, list):
                    args.predictors_to_be_added = [args.predictors_to_be_added]

                if 'SemanticContainer' not in args.predictors_to_be_added:
                    args.predictors_to_be_added.append('SemanticContainer')

        if args.use_attr:
            add_predictor(args)
        

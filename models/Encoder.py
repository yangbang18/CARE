import torch
import torch.nn as nn
from config import Constants
from models.components.activations import get_activation
from models.components.Embeddings import NaiveEmbeddings
from .Att_Encoder import *


__all__ = (
    'SingleStreamEmbedder',
    'Embedder',
    'EncoderWithHighWayBN',
    'MultiTransformerEncoder',
    'TransformerEncoder',
)


def get_encoder(opt: dict) -> nn.Module:
    class_name = opt['encoder']
    if class_name not in globals():
        raise ValueError('We can not find the class `{}` in {}'.format(class_name, __file__))

    return globals()[class_name](opt)


##############################
#       Basic Classes        #
##############################
class SingleStream(nn.Module):
    def __init__(self, opt, module_func):
        super().__init__()
        input_dim = sum([opt.get('dim_' + char, None) for char in opt['modality'].lower()])
        output_dim = opt.get('dim_hidden', 512)
        dropout = opt.get('encoder_dropout_prob', 0.5)
        self.encoder = module_func(input_dim, output_dim, dropout, opt)

    def forward(self, input_feats: torch.Tensor, **kwargs) -> dict:
        if hasattr(self, 'pre_processing'):
            input_feats = self.pre_processing(input_feats)
        
        input_feats = torch.cat(input_feats, dim=-1)
        encoder_hidden_states = self.encoder(input_feats)
        results = {'encoder_hidden_states': encoder_hidden_states}

        if hasattr(self, 'post_processing'):
            results.update(self.post_processing(results['encoder_hidden_states']))

        return results


class MultipleStreams(nn.Module):
    def __init__(self, opt, module_func, is_rnn=False, check_valid=False):
        super().__init__()
        self.encoders = []
        modality = opt['modality'].lower()
        for char in modality:
            if char == 't':
                # for processing retrieved captions
                module = Text_Embedder(opt)
            else:
                input_dim = opt.get('dim_' + char, None)
                output_dim = opt.get('dim_hidden', 512)
                dropout = opt.get('encoder_dropout_prob', 0.5)
                assert input_dim is not None, \
                    'The modality is {}, but dim_{} can not be found in opt'.format(modality, char)
                
                module = module_func(input_dim, output_dim, dropout, opt)

            self.add_module("Encoder_%s" % char.upper(), module)
            self.encoders.append(module)
 
        self.num_feats = len(modality)
        self.is_rnn = is_rnn

        self.fusion_type = opt.get('fusion', 'temporal_concat')
        supported_fusion_type = ['temporal_concat', 'addition', 'none', 'channel_concat']
        if check_valid and self.fusion_type not in supported_fusion_type:
            raise ValueError('We now only support the fusion_type: {}'.format(supported_fusion_type))
        
        self.modality = modality
        self.modality_for_decoder = opt.get('modality_for_decoder', modality)
        self.modality_for_predictor = opt.get('modality_for_predictor', modality)
        self.opt = opt

    def forward(self, input_feats: torch.Tensor, **kwargs) -> dict:
        assert self.num_feats == len(input_feats)

        if hasattr(self, 'pre_processing'):
            input_feats = self.pre_processing(input_feats)

        if not self.is_rnn:
            encoder_hidden_states = []
            for char, encoder, feats in zip(self.modality, self.encoders, input_feats):
                if char == 't':
                    this_hidden_states = encoder(feats, kwargs['Embeddings'])
                    ret_input_ids = feats
                    ret_text_embs = this_hidden_states
                else:
                    this_hidden_states = encoder(feats)
                encoder_hidden_states.append(this_hidden_states)
        else:
            # TODO
            pass
        
        data = {'encoder_hidden_states': encoder_hidden_states}
        data['mean_encoder_hidden_states'] = [item.mean(1) for item in encoder_hidden_states]

        self.prepare_inputs_for_components(data, self.modality_for_predictor, 'inputs_for_predictor')
        self.prepare_inputs_for_components(data, self.modality_for_decoder, 'inputs_for_decoder')
        
        if 't' in self.modality:
            if 'inputs_for_decoder' in data:
                data['inputs_for_decoder']['ret_input_ids'] = ret_input_ids
                data['inputs_for_decoder']['ret_text_embs'] = ret_text_embs
            else:
                data['ret_input_ids'] = ret_input_ids
                data['ret_text_embs'] = ret_text_embs
            index = self.modality.index('t')
            _ = data['encoder_hidden_states'].pop(index)
            _ = data['mean_encoder_hidden_states'].pop(index)

        data.update(self.post_processing(data))
        return data
    
    def prepare_inputs_for_components(self, data, component_modality, key_name):
        if component_modality and component_modality != self.modality:
            assert 't' not in component_modality
            new_data = {}
            for k, v in data.items():
                if isinstance(v, dict):
                    continue
                assert isinstance(v, (tuple, list))
                assert len(v) == len(self.modality)
                new_v = [item for char, item in zip(self.modality, v) if char in component_modality]
                new_data[k] = new_v
            
            new_data.update(self.post_processing(new_data))
            data[key_name] = new_data

    def post_processing(self, data: dict) -> dict:
        encoder_hidden_states = data['encoder_hidden_states']

        if self.fusion_type != 'none':
            if not isinstance(encoder_hidden_states, list):
                encoder_hidden_states = [encoder_hidden_states]
            if self.fusion_type == 'addition':
                encoder_hidden_states = torch.stack(encoder_hidden_states, dim=0).mean(0)
            elif self.fusion_type == 'temporal_concat':
                encoder_hidden_states = torch.cat(encoder_hidden_states, dim=1)
            elif self.fusion_type == 'channel_concat':
                encoder_hidden_states = torch.cat(encoder_hidden_states, dim=2)
        
        return {'encoder_hidden_states': encoder_hidden_states}


##############################
#   Derived Encoder Classes  #
##############################
class SingleStreamEmbedder(SingleStream):
    def __init__(self, opt):
        module_func = lambda x,y,z,opt: nn.Sequential(nn.Linear(x, y), nn.LayerNorm(y, eps=opt['layer_norm_eps']), nn.Dropout(z))
        super().__init__(opt, module_func)


class Embedder(MultipleStreams):
    def __init__(self, opt):
        module_func = lambda x,y,z,opt: nn.Sequential(nn.Linear(x, y), nn.LayerNorm(y, eps=opt['layer_norm_eps']), nn.Dropout(z))
        super().__init__(opt, module_func, check_valid=True)


class ReLUEmbedder(MultipleStreams):
    def __init__(self, opt):
        module_func = lambda x,y,z,opt: nn.Sequential(nn.Linear(x, y), nn.ReLU(), nn.Dropout(z))
        super().__init__(opt, module_func, check_valid=True)


class Identity(MultipleStreams):
    def __init__(self, opt):
        assert len(opt['modality']) == 1, opt['modality']
        module_func = lambda x,y,z,opt: nn.Identity()
        super().__init__(opt, module_func)


class EncoderWithHighWayBN(MultipleStreams):
    def __init__(self, opt):
        module_func = lambda x,y,z,opt: nn.Sequential(nn.Linear(x, y), HighWay(y), BN1d(y), nn.Dropout(z))
        super().__init__(opt, module_func, check_valid=True)


class MultiTransformerEncoder(MultipleStreams):
    def __init__(self, opt):
        module_func = lambda x,y,z,opt: nn.Sequential(nn.Linear(x, y), TransformerEncoderBase(opt))
        super().__init__(opt, module_func, check_valid=True)


class TransformerEncoder(MultipleStreams):
    def __init__(self, opt):
        # module_func = lambda x,y,z,opt: nn.Sequential(nn.Linear(x, y), nn.Dropout(z))
        module_func = lambda x,y,z,opt: nn.Linear(x, y)
        super().__init__(opt, module_func, check_valid=True)
        self.backbone = TransformerEncoderBase(opt)
    
    def post_processing(self, encoder_hidden_states: torch.Tensor) -> dict:
        return self.backbone(encoder_hidden_states, only_return_encoder_hidden_states=False)


##############################
#       Basic Components     #
##############################
class HighWay(nn.Module):
    def __init__(self, hidden_size, with_gate=True):
        super().__init__()
        self.with_gate = with_gate
        self.w1 = nn.Linear(hidden_size, hidden_size)
        if self.with_gate:
            self.w2 = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        #self._init_weights()

    def forward(self, x):
        y = self.tanh(self.w1(x))
        if self.with_gate:
            gate = torch.sigmoid(self.w2(x))
            return gate * x + (1 - gate) * y
        else:
            return x + y


class BN1d(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.bn = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        if len(x.shape) == 2:
            return self.bn(x)
        else:
            assert x.shape[-1] == self.hidden_size
            rest_shape = x.shape[:-1]
            return self.bn(x.contiguous().view(-1, self.hidden_size)).view(*rest_shape, self.hidden_size)


class TransformerEncoderBase(nn.Module):
    def __init__(self, opt):
        super().__init__()
        from models.components.Embeddings import PositionalEmbedding
        from models.components.Layers import EncoderLayer

        self.trainable_pe = opt.get('trainable_pe', False)
        if self.trainable_pe:
            self.position_embeddings = nn.Embedding(opt['n_frames'], opt['dim_hidden'])
        else:
            self.position_embeddings = PositionalEmbedding(opt['n_frames'], opt['dim_hidden'])
        
        self.LayerNorm = nn.LayerNorm(opt['dim_hidden'], eps=opt['layer_norm_eps'])
        self.dropout = nn.Dropout(opt['hidden_dropout_prob'])

        self.layers = nn.ModuleList([EncoderLayer(opt) for _ in range(opt['num_hidden_layers_encoder'])])
    
    def forward(self, input_feats, only_return_encoder_hidden_states=True):
        if not isinstance(input_feats, list):
            input_feats = [input_feats]

        if self.trainable_pe:
            seq_length = input_feats[0].size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_feats[0].device)
            position_ids = position_ids.unsqueeze(0).repeat(input_feats[0].size(0), 1)
            position_embeddings = self.position_embeddings(position_ids)
        else:
            position_embeddings = self.position_embeddings(input_feats[0])

        hidden_states = [feats + position_embeddings for feats in input_feats]
        hidden_states = torch.cat(hidden_states, dim=1)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        all_encoder_hidden_states = [hidden_states]
        all_encoder_intra_attentions = ()

        for layer in self.layers:
            hidden_states, intra_attention_probs, _ = layer(
                hidden_states=all_encoder_hidden_states[-1], 
                attention_mask=None,
                head_mask=None,
            )

            all_encoder_hidden_states.append(hidden_states)
            all_encoder_intra_attentions = all_encoder_intra_attentions + (intra_attention_probs, )

        if only_return_encoder_hidden_states:
            return all_encoder_hidden_states[-1]
            
        return {
            'encoder_hidden_states': all_encoder_hidden_states[-1],
            'all_encoder_hidden_states': all_encoder_hidden_states,
            'all_encoder_intra_attentions': all_encoder_intra_attentions,
        }


class LightCNN(nn.Module):
    def __init__(self, chs = [12, 32, 128, 512], resolution = 7, dropout_rate=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(chs[0], chs[1], kernel_size=3, stride=1, padding=0) # (N, 12, 7, 7) --> (N, 32, 5, 5)
        self.bn1 = nn.BatchNorm2d(chs[1])
        self.conv2 = nn.Conv2d(chs[1], chs[2], kernel_size=3, stride=1, padding=0) # (N, 32, 5, 5) --> (N, 128, 3, 3)
        self.bn2 = nn.BatchNorm2d(chs[2])
        self.conv3 = nn.Conv2d(chs[2], chs[3], kernel_size=3, stride=1, padding=0) # (N, 128, 3, 3) --> (N, 512, 1, 1)
        self.bn3 = nn.BatchNorm2d(chs[3])
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.resolution = resolution

    def forward(self, x):
        assert x.dim() == 3
        bsz, n_frames = x.shape[:2]
        x = x.view(bsz * n_frames, -1, self.resolution, self.resolution)

        for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
            x = self.relu(bn(conv(x)))

        x = x.view(bsz, n_frames, -1)
        return self.dropout(x)


class POS_Layer(nn.Module):
    def __init__(self, resolution = 7):
        super().__init__()
        self.pos_bias = nn.Parameter(torch.zeros(resolution**2))
        self.resolution = resolution

    def forward(self, x):
        assert x.dim() == 3
        bsz, n_frames = x.shape[:2]
        x = x.view(bsz * n_frames, -1, self.resolution**2)
        x = x + self.pos_bias
        x = x.view(bsz, n_frames, -1)
        return x


class Text_Embedder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        if self.opt.get('has_retrieval_embs', False):
            self.embs = NaiveEmbeddings(opt['vocab_size'], opt['max_len'], opt['dim_hidden'], padding_idx=Constants.PAD)
        
        if self.opt.get('has_retrieval_rnn', False):
            self.rnn = nn.LSTM(
               input_size=opt['dim_hidden'],
               hidden_size=opt['dim_hidden'],
               batch_first=True, 
               bidirectional=True,
            )
            self.LayerNorm = nn.LayerNorm(opt['dim_hidden'], eps=opt['layer_norm_eps'])
            self.dropout = nn.Dropout(0.5)
    
    def forward(self, input_ids, Embeddings):
        assert input_ids.dim() == 3, input_ids.shape
        bsz, n_retrieval, max_len = input_ids.shape
        input_ids = input_ids.view(bsz * n_retrieval, max_len)

        if hasattr(self, 'embs'):
            embs = self.embs(input_ids)
        else:
            embs, *_ = Embeddings(input_ids, only_word_and_position=True) # (bsz * n_retrieval, max_len, dim_hidden)
        
        if hasattr(self, 'rnn'):
            embs, _ = self.rnn(embs)
            embs = embs.split(self.opt['dim_hidden'], 2)
            embs = (embs[0] + embs[1]) / 2
            embs = self.dropout(self.LayerNorm(embs))

        embs = embs.view(bsz, n_retrieval, max_len, -1) # (bsz, n_retrieval, max_len, dim_hidden)
        return embs


class VOE(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.rnns = []
        for i, char in enumerate(opt['modality']):
            input_size = opt['dim_%s'%char] + (opt['dim_hidden'] if i else 0)
            rnn = nn.GRU(input_size, opt['dim_hidden'], batch_first=True, num_layers=1)
            self.add_module("RNN_%s" % char, rnn)
            self.rnns.append(rnn)
        
        self.bn = BN1d(opt['dim_hidden'])
        self.dropout = nn.Dropout(opt.get('encoder_dropout_prob', 0.5))

    def forward(self, input_feats, **kwargs):
        assert len(input_feats) == len(self.rnns)
        
        n_frames = input_feats[0].shape[1]
        assert all(input_f.size(1) == n_frames for input_f in input_feats)

        rnn_hidden_states = None
        for i in range(len(input_feats)):
            if i:
                inputs = torch.cat([self.dropout(rnn_outputs), input_feats[i]], dim=2)
            else:
                inputs = input_feats[i]

            rnn_outputs, rnn_hidden_states = self.rnns[i](inputs, rnn_hidden_states)

        rnn_outputs = self.bn(rnn_outputs)

        return {
            'encoder_hidden_states': rnn_outputs,
            'mean_encoder_hidden_states': [rnn_outputs.mean(1)],
        }

import torch
import torch.nn as nn


class Predictor_length(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.net = nn.Sequential(
                    nn.Linear(opt['dim_hidden'], opt['dim_hidden']),
                    nn.ReLU(),
                    nn.Dropout(opt['hidden_dropout_prob']),
                    nn.Linear(opt['dim_hidden'], opt['max_len']),
                )

    def forward(self, encoder_hidden_states, **kwargs):
        if isinstance(encoder_hidden_states, list):
            assert len(encoder_hidden_states) == 1
            encoder_hidden_states = encoder_hidden_states[0]
        assert len(encoder_hidden_states.shape) == 3

        out = self.net(encoder_hidden_states.mean(1))
        return {'preds_length': torch.log_softmax(out, dim=-1)}
    
    @staticmethod
    def add_specific_args(parent_parser: object) -> object:
        parser = parent_parser.add_argument_group(title='Length Prediction Settings')
        parser.add_argument('--length_prediction', default=False, action='store_true')
        parser.add_argument('--length_prediction_scale', type=float, default=1.0)
        return parent_parser
    
    @staticmethod
    def check_args(args: object) -> None:
        if args.length_prediction:
            if not isinstance(args.crits, list):
                args.crits = [args.crits]
            if 'length' not in args.crits:
                args.crits.append('length')

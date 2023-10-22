from .RNN_single_layer import SingleLayerRNNDecoder, VOERNNDecoder
from .RNN_multi_layers import TopDownAttentionRNNDecoder
from .Transformer import (
    TransformerDecoder, 
    TwoStageTransformerDecoder, 
)

def get_decoder(opt: dict):
    class_name = opt['decoder']
    if class_name not in globals():
        raise ValueError('We can not find the class `{}` in {}'.format(class_name, __file__))

    return globals()[class_name](opt)

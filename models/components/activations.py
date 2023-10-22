from torch import nn

ACT2FN = {
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
    "tanh": nn.Tanh(),
    "linear": nn.Identity(),
    "sigmoid": nn.Sigmoid(),
    "leakyrelu": nn.LeakyReLU(),
}

def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError(f"function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}")

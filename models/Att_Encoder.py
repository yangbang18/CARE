import torch
import torch.nn as nn
from config import Constants


class CNNBase(nn.Module):
    def __init__(self, dim_input, dim_hidden, n_frames, layer_norm_eps=1e-12, 
                kernel_size=(3, 3, 3), padding=(1, 1, 1)
    ):
        super().__init__()
        '''
            shape of input: (bsz, n_frames, n_layers, n_patches)
                        --> (bsz, 1, n_frames, ws, ws)
        '''
        
        self.n_patches = dim_input
        self.window_size = int(dim_input**0.5)
        self.dim_hidden = dim_hidden
        self.n_frames = n_frames

        assert self.window_size**2 == self.n_patches, self.n_patches
        
        
        self.block1 = nn.Sequential(
            nn.Conv3d(1, 2, kernel_size=kernel_size, padding=padding, stride=1),
            nn.BatchNorm3d(2),
            nn.ReLU(),
        )
        self.pool1 = nn.AvgPool3d(kernel_size=(2,1,1), stride=(2,1,1), padding=0)

        self.block2 = nn.Sequential(
            nn.Conv3d(2, 4, kernel_size=kernel_size, padding=padding, stride=1),
            nn.BatchNorm3d(4),
            nn.ReLU(),
        )
        self.pool2 = nn.AvgPool3d(kernel_size=(2,1,1), stride=(2,1,1), padding=0)

        kernel_size = (n_frames // 4, ) + kernel_size[1:]
        padding = (0, ) + padding[1:]
        self.block3 = nn.Sequential(
            nn.Conv3d(4, 8, kernel_size=kernel_size, padding=padding, stride=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )

        self.net = nn.Linear(self.n_patches * 8, dim_hidden)
        self.LN = nn.LayerNorm(dim_hidden, eps=layer_norm_eps)
    
    def forward(self, x):
        N = x.size(0)
        x = x.mean(2) # (bsz, n_frames, n_layers, n_patches) -> (bsz, n_frames, n_patches)
        x = x.view(-1, 1, self.n_frames, self.window_size, self.window_size)

        # (bsz, 1, n_frames, ws, ws) -> (bsz, 2, n_frames//2, ws, ws)
        x = self.pool1(self.block1(x))
        # (bsz, 2, n_frames//2, ws, ws) -> (bsz, 4, n_frames//4, ws, ws)
        x = self.pool2(self.block2(x))
        # (bsz, 4, n_frames//4, ws, ws) -> (bsz, 8, 1, ws, ws)
        x = self.block3(x) 

        x = x.view(N, 1, -1)
        x = self.LN(self.net(x))
        return x


class CNN1(CNNBase):
    def __init__(self, opt):
        super().__init__(
            dim_input=opt['dim_t'], 
            dim_hidden=opt['dim_hidden'], 
            n_frames=opt['n_frames'], 
            layer_norm_eps=opt['layer_norm_eps'], 
            kernel_size=(3, 3, 3), 
            padding=(1, 1, 1),
        )


class CNN2(CNNBase):
    def __init__(self, opt):
        super().__init__(
            dim_input=opt['dim_t'], 
            dim_hidden=opt['dim_hidden'], 
            n_frames=opt['n_frames'], 
            layer_norm_eps=opt['layer_norm_eps'], 
            kernel_size=(7, 3, 3), 
            padding=(3, 1, 1),
        )


class CNN3(CNNBase):
    def __init__(self, opt):
        super().__init__(
            dim_input=opt['dim_t'], 
            dim_hidden=opt['dim_hidden'], 
            n_frames=opt['n_frames'], 
            layer_norm_eps=opt['layer_norm_eps'], 
            kernel_size=(7, 5, 5), 
            padding=(3, 2, 2),
        )


# class SortNet(nn.Module):
#     def __init__(self, dim_input, dim_hidden, n_frames, layer_norm_eps=1e-12, 
#                 kernel_size=(3, 3, 3), padding=(1, 1, 1)
#     ):
#         super().__init__()
#         '''
#             shape of input: (bsz, n_frames, n_layers, n_patches)
#                         --> (bsz, 1, n_frames, ws, ws)
#         '''
        
#         self.n_patches = dim_input
#         self.window_size = int(dim_input**0.5)
#         self.dim_hidden = dim_hidden
#         self.n_frames = n_frames

        
#         self.net = nn.Linear(self.n_patches * 8, dim_hidden)
#         self.LN = nn.LayerNorm(dim_hidden, eps=layer_norm_eps)
    
#     def forward(self, x):
#         N = x.size(0)
#         x = x.mean(2) # (bsz, n_frames, n_layers, n_patches) -> (bsz, n_frames, n_patches)
#         x = x.view(-1, 1, self.n_frames, self.window_size, self.window_size)

#         # (bsz, 1, n_frames, ws, ws) -> (bsz, 2, n_frames//2, ws, ws)
#         x = self.pool1(self.block1(x))
#         # (bsz, 2, n_frames//2, ws, ws) -> (bsz, 4, n_frames//4, ws, ws)
#         x = self.pool2(self.block2(x))
#         # (bsz, 4, n_frames//4, ws, ws) -> (bsz, 8, 1, ws, ws)
#         x = self.block3(x) 

#         x = x.view(N, -1)
#         x = self.LN(self.net(x))
#         return x


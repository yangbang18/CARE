import torch.nn as nn
from typing import Optional


def get_backbone(opt: dict) -> Optional[nn.Module]:
    backbones, preprocess_func = [], []
    if opt.get('with_backbones', []):
        backbone_names = [item.strip() for item in opt['with_backbones']]

        assert len(backbone_names) == len(opt['modality']), \
            f"please specify backbones for each modality, recevied {len(backbone_names)} backbone names and {len(opt['modality'])} modalities"

        for name, char in zip(backbone_names, opt['modality']):
            print(f'- The backbone for the modality {char} is {name}')
            
            if char != 'i':
                assert not name, "only support specifying the image backbone right now"
                backbone = preprocess = None
            else:
                if 'clip' in name:
                    import CLIP.clip as clip
                    arch = name.split('~')[1]
                    backbone, preprocess = clip.load(arch, device='cpu', jit=False)
                    backbone = backbone.visual # we only need the visual encoder
                else:
                    import pretrainedmodels
                    from pretrainedmodels import utils
                    assert hasattr(pretrainedmodels, name), \
                        f'Can not find {name} in pretrainedmodels'
                    
                    backbone = getattr(pretrainedmodels, name)(pretrained='imagenet')
                    preprocess = utils.LoadTransformImage(backbone)
                    backbone.last_linear = utils.Identity()
            
            backbones.append(backbone)
            preprocess_func.append(preprocess)

    if not len(backbones):
        return None

    return BackboneManager(opt, backbones, preprocess_func)


class BackboneManager(nn.Module):
    def __init__(self, opt, backbones, preprocess_func):
        super().__init__()
        self.backbones = backbones
        self.preprocess_func = preprocess_func
        self.modality2index = {}
        for _, (backbone, char) in enumerate(zip(backbones, opt['modality'])):
            if backbone is None:
                continue
            self.add_module('Backbone_{}'.format(char.upper()), backbone)
            self.modality2index[char] = _

    def forward(self, feats):
        outputs = []
        for backbone, f in zip(self.backbones, feats):
            if backbone is None:
                outputs.append(f)
            else:
                bsz, n_frames, *rest_shape = f.shape
                f = f.view(bsz * n_frames, *rest_shape)
                out = backbone(f)
                out = out.view(bsz, n_frames, -1)
                outputs.append(out)
        return outputs

    def get_preprocess_func(self, modality):
        if modality not in self.modality2index:
            return None
        return self.preprocess_func[self.modality2index[modality]]
    
    def get_backbone(self, modality):
        if modality not in self.modality2index:
            return None
        return self.backbones[self.modality2index[modality]]

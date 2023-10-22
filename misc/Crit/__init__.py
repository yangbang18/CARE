from . import prepare
from .base import Criterion
import copy


def get_crit_info(opt, crit):
    func_name = f'get_crit_info_{crit}'

    if not hasattr(prepare, func_name):
        raise ModuleNotFoundError(
            f'\nWe can not find the function `{func_name}` in misc.Crit.prepare!\n'
            'Given `crit` = X, you need to:\n'
            '1) implement the function `get_crit_info_X` in misc.Crit.prepare.\n'
            '2) feel free to implement a criterion in a separated file, e.g., misc.Crit.crit_X.\n'
            '3) examine wheter the captioner can already give predictions for X.\n'
            '   if not, you need to implement the predictor in models.Predictor.pred_X.'
        )

    return getattr(prepare, func_name)(opt)


def get_criterion(opt, skip_crit_list=[], override_opt={}):
    if len(override_opt):
        _opt = copy.deepcopy(opt)
        _opt.update(override_opt)
    else:
        _opt = opt

    assert isinstance(_opt['crits'], list)
    satisfied_crits = [item for item in _opt['crits'] if item not in skip_crit_list]

    crit_objects = []
    names = []
    scales = []
    for crit in satisfied_crits:
        info = get_crit_info(_opt, crit)
        
        if info is None:
            raise ModuleNotFoundError(
                f'- '
            )
        
        assert len(info) == 3
        pre_len = None
        for _this, _all in zip(info, [crit_objects, names, scales]):
            if not isinstance(_this, list):
                _this = [_this]

            if pre_len is None:
                pre_len = len(_this)
            else:
                assert pre_len == len(_this), \
                    f'- (object_func, name, scale) of {crit} do not have the same number of elements!'

            _all.extend(_this)
    
    if not len(crit_objects):
        return None

    return Criterion(
            crit_objects=crit_objects,
            names=names,
            scales=scales,
        )
        
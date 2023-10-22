from typing import List, Dict, Tuple, Union
import torch
from ..logger import AverageMeter


class CritBase(object):
    def __init__(self, 
            keys: List[str], 
            weights: Union[List[float], float] = 1.0, 
            batch_mean: bool = True
        ):
        super(CritBase, self).__init__()
        self.keys = keys
        self.weights = weights
        self.batch_mean = batch_mean

    def _step(self, *inputs) -> torch.Tensor:
        raise NotImplementedError()

    def __call__(self, kwargs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, float]:
        sources1, sources2, *others = [kwargs.get(key, None) for key in self.keys]
        
        if not isinstance(sources1, list):
            # assert type(sources1) is torch.Tensor
            sources1 = [sources1]
        
        if not isinstance(sources2, list):
            # assert type(sources2) is torch.Tensor
            sources2 = [sources2] * len(sources1)
        else:
            assert len(sources1) == len(sources2), f'length of sources1: {len(sources1)}\tlength of sources2: {len(sources2)}'

        if not isinstance(self.weights, list):
            self.weights = [self.weights] * len(sources1)

        assert len(sources1) == len(self.weights)

        loss = None
        dinominator = float(sources1[0].size(0)) if self.batch_mean else 1.0

        for i, (weight, src1, src2) in enumerate(zip(self.weights, sources1, sources2)):
            if loss is None:
                loss = weight * self._step(i, src1, src2, *others) / dinominator
            else:
                loss = loss + weight * self._step(i, src1, src2, *others) / dinominator
        
        return loss, dinominator


class Criterion(object):
    """
        Calculating losses or some metrics for all tasks

        Standard operations:
            1. before a epoch, Criterion.reset_loss_recorder()
            2. during a epoch, Criterion.get_loss(forward_results)
            3. after  a epoch, Criterion.get_loss_info()
    """ 
    def __init__(self, crit_objects, names, scales):
        assert len(crit_objects) == len(names)
        assert len(names) == len(scales)
        self.crit_objects = crit_objects
        self.num_loss = len(crit_objects)
        self.names = names
        self.scales = scales
        self.n_current_round = 0

        self.reset_loss_recorder()
    
    def set_scales(self, new_scales):
        assert len(new_scales) == len(self.scales)
        self.scales = new_scales
        
    def reset_loss_recorder(self):
        self.loss_recorder = [AverageMeter() for _ in range(self.num_loss)]
        for crit_object in self.crit_objects:
            if getattr(crit_object, 'reset_recorder', None) is not None:
                crit_object.reset_recorder()

    def get_loss(self, results, **kwargs):
        """
            args:
                results: dict, contains the forward results of the model and some ground-truths
        """
        loss = []
        for i in range(self.num_loss):
            # calculate the i-th loss
            assert isinstance(self.crit_objects[i], CritBase)
            i_loss, num_samples = self.crit_objects[i](results)
            
            # weighting the i-th loss
            loss.append(i_loss * self.scales[i])

            # update the statistics of the i-th loss
            self.loss_recorder[i].update(i_loss.item(), num_samples)

        # loss = loss1 * scale1 + loss2 * scale2 + ... 
        loss = torch.stack(loss, dim=0).sum(0)
        return loss

    def get_loss_info(self):
        all_names = self.names.copy()
        all_info = [meter.avg for meter in self.loss_recorder]

        for crit_object in self.crit_objects:
            if getattr(crit_object, 'get_info', None) is not None:
                this_name, this_info = crit_object.get_info()
                all_names += this_name
                all_info += this_info

        # e.g., ['Cap Loss', 'Word Acc0', 'Perplexity'], [31.8, 0.385, 53.0]
        # return all_names, all_info 
        return {n: i for n, i in zip(all_names, all_info)}

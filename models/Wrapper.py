import torch
from pytorch_lightning import LightningModule
from typing import List, Dict, Any, Optional, Tuple, Union

from models.Framework import get_framework
from misc.Crit import get_criterion
from models.Translator import get_translator

import pickle, json, os
import time
from tqdm import tqdm
from collections import defaultdict
from misc.cocoeval import COCOScorer, suppress_stdout_stderr
from misc.utils import (
    to_sentence, 
    to_sentence_with_tokenizer,
    filter_weight_decay, 
    save_dict_to_csv, 
    analyze_length_novel_unique
)


class ModelBase(LightningModule):
    def __init__(self, opt: Dict[str, Any], new_opt_used_to_override: Dict[str, Any] = {}):
        super().__init__()
        # passed arguments (hyperparameters) will be saved, we can assess it by `self.hparams`
        self.save_hyperparameters()
        
        newest_opt = {**self.hparams.opt, **self.hparams.new_opt_used_to_override}

        # captioning model
        self.captioner = get_framework(newest_opt)

        # translator aims to generate captions from scratch with specificed decoding algorithms, 
        # e.g., greedy search (beam size = 1), beam search (beam size > 1), etc
        self.translator = get_translator(newest_opt)

        self.tokenizer = newest_opt.get('tokenizer', None)
        self.coco_eval = 'lang' in newest_opt['crits']

    def training_step(self, batch, batch_idx):
        raise NotImplementedError('Please implement the function `training_epoch_end` in the derived class')
    
    def validation_step(self, batch, batch_idx):
        if self.coco_eval:
            return self.translate_step(batch, vocab=self.get_vocab(), assert_only_a_caption_per_video=True)
        else:
            assert not isinstance(self.captioner, (tuple, list))
            assert self.eval_criterion is not None
            feedforward_outputs = self.captioner.feedforward_step(batch)
            self.eval_criterion.get_loss({**feedforward_outputs, **batch})
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def training_epoch_end(self, all_step_outputs) -> None:
        raise NotImplementedError('Please implement the function `training_epoch_end` in the derived class')
    
    def validation_epoch_end(self, all_step_outputs, crit_prefix='vali', log_best=True) -> None:
        if self.coco_eval:
            self.evaluation(all_step_outputs, references=self.get_references(), 
                log_scores=True, log_best_ever_scores=True, crit_prefix=crit_prefix)
        else:
            loss_info = self.eval_criterion.get_loss_info()
            self.log_dict({'{}_{}'.format(crit_prefix, k): v for k, v in loss_info.items()})
            
            if log_best and 'mAP' in loss_info:
                if not hasattr(self, 'best_mAP') or loss_info['mAP'] > self.best_mAP:
                    self.best_mAP = loss_info['mAP']
                
                self.log('best_mAP', self.best_mAP, prog_bar=True)
            
            self.eval_criterion.reset_loss_recorder()

    def test_epoch_end(self, 
            all_step_outputs, 
            log_scores=True, 
            verbose=True, 
            keys_added_to_scores=['seed'],
            analyze=True, 
            save_csv_path=None,
        ):
        if not self.coco_eval:
            return self.validation_epoch_end(all_step_outputs, 'test', log_best=False)

        opt = self.get_opt()

        # check if there is exact one caption per video
        tmp_vid = list(all_step_outputs[0].keys())[0]
        n_caption_per_video = len(all_step_outputs[0][tmp_vid])

        if n_caption_per_video == 1:
            # if there is one caption per video, coco evaluation can be run without errors
            preds_for_completion = {}
            if opt['dataset'] == 'VATEX' and opt.get('feats', '') != 'I3D':
                print('- Evaluating on the VATEX dataset')
                if opt.get('VATEX_I3D_preds_json', ''):
                    preds_for_completion = json.load(open(opt['VATEX_I3D_preds_json'], 'rb'))
                    print('- Loading I3D predictions from', opt['VATEX_I3D_preds_json'])
                else:
                    print('- Partial data is missing, only obtain the subset\'s performance')

            scores, detail_scores, pred_captions = self.evaluation(all_step_outputs, references=self.get_references(),
                                                    log_scores=log_scores, log_prefix='test', crit_prefix='test',
                                                    preds_for_completion=preds_for_completion)
        else:
            print(f'- We do not run coco evaluation because each video has {n_caption_per_video} generated captions.')
            print('- If you want to save generated captions, please remember to pass `--json_path $path_to_save --json_name $filename_to_save` arguments')
            scores, detail_scores = {}, None
            pred_captions = {}
            for item in all_step_outputs:
                pred_captions.update(item)
        
        for key in keys_added_to_scores:
            if isinstance(opt[key], (tuple, list)):
                value = '-'.join([str(item) for item in opt[key]])
            else:
                value = opt[key]
            scores[key] = value

        if analyze:
            info_corpus = self.get_info_corpus()
            ave_length, novel, unique, usage = analyze_length_novel_unique(
                info_corpus['captions'], 
                pred_captions, 
                vocab=self.get_vocab(), 
                splits=info_corpus['info']['split'], 
                n=1
            )
            scores.update({'ave_length': ave_length, 'novel': novel, 'unique': unique, 'usage': usage})

        if opt.get("save_csv", False):
            save_csv_path = opt['checkpoint_path'] if save_csv_path is None else save_csv_path
            save_dict_to_csv(save_csv_path, opt.get('csv_name', 'test_result.csv'), scores)
        
        if opt.get('json_path', ''):
            assert 'json_name' in opt.keys()
            os.makedirs(opt['json_path'], exist_ok=True)
            save_path = os.path.join(opt['json_path'], opt['json_name'])
            json.dump(pred_captions, open(save_path, 'w'))
        
        if verbose:
            for k, v in scores.items():
                if isinstance(v, str):
                    tqdm.write(k + ': %s' % v)
                else:
                    tqdm.write(k + ': %g' % v)
        
        return scores, detail_scores, pred_captions
    
    def forward(self, batch, **kwargs) -> Dict[str, List[dict]]:
        # in lightning, forward defines the prediction/inference actions
        vocab = kwargs.pop('vocab', None)
        if vocab is None:
            vocab = self.get_vocab()
        return self.translate_step(batch, vocab=vocab, **kwargs)
    
    def translate_step(self, 
            batch: Dict[str, Any], 
            vocab: Dict[int, str], 
            assert_only_a_caption_per_video=False, 
            verbose=False,
            inference_latency=False,
        ) -> Dict[str, List[dict]]:

        if hasattr(self, 'preprocess_batch_before_translate_step'):
            self.preprocess_batch_before_translate_step(batch)

        # Model ensembling is achieved by the translator
        if not isinstance(self.captioner, list):
            models = [self.captioner]
        else:
            models = self.captioner

        hyps_of_a_batch, scores_of_a_batch = self.translator.translate_batch(
            models=models, 
            batch=batch, 
            vocab=self.get_vocab(), 
            teacher_model_wrapper=getattr(self, 'teacher_model_wrapper', None)
        )

        if not inference_latency and len(models) == 1 and getattr(self, 'eval_criterion', None) is not None:
            feedforward_outputs = self.captioner.feedforward_step(batch)
            self.eval_criterion.get_loss({**feedforward_outputs, **batch})

        bsz = len(hyps_of_a_batch)
        preds_of_a_batch = defaultdict(list)
        for i in range(bsz):
            hyps_of_a_video = hyps_of_a_batch[i]
            scores_of_a_video = scores_of_a_batch[i]
            video_id = batch['video_ids'][i]

            assert isinstance(hyps_of_a_video, list)
            if assert_only_a_caption_per_video:
               assert len(hyps_of_a_video) == 1

            for hyp, score in zip(hyps_of_a_video, scores_of_a_video):
                if self.tokenizer is None:
                    caption = to_sentence(hyp, vocab)
                else:
                    caption = to_sentence_with_tokenizer(hyp, self.tokenizer)
                
                if verbose:
                    tqdm.write('{}: {}({})'.format(video_id, caption, score))

                preds_of_a_batch[video_id].append({
                    'image_id': video_id, 
                    'caption': caption, 
                    'score': score
                })

        return preds_of_a_batch
    
    def evaluation(self, 
            all_step_outputs: Dict[str, List[dict]], 
            references: Dict[str, List[dict]], 
            scorer: object = COCOScorer(),
            log_scores: bool = True,
            log_best_ever_scores: bool = False,
            log_prefix: str = '',
            crit_prefix: str = '',
            preds_for_completion: Dict[str, List[dict]] = {},
        ) -> Tuple[Dict[str, Any], Dict[str, Any]]:

        preds = {}
        for item in all_step_outputs:
            preds.update(item)
        
        if len(preds_for_completion):
            num_missing = 0
            for key in preds_for_completion:
                if key not in preds:
                    num_missing += 1
                    preds[key] = preds_for_completion[key]
            
            print(f'- Adding {num_missing} missing predictions for evaluation')
        
        with suppress_stdout_stderr():
            scores, detail_scores = scorer.score(references, preds, preds.keys())

        candidate_scores = [scores['Bleu_4'], scores['METEOR'], scores['ROUGE_L'], scores['CIDEr']]
        scores['Sum'] = sum([score for (score, flag) in zip(candidate_scores, self.hparams.opt['metric_sum']) if flag])

        loss_info = None
        if getattr(self, 'eval_criterion', None) is not None:
            loss_info = self.eval_criterion.get_loss_info()
            if 'mAP' in loss_info:
                scores['mAP'] = loss_info.pop('mAP')

        if log_scores:
            renamed_scores = {'{}_{}'.format(log_prefix, k): v for k, v in scores.items()} if log_prefix else scores
            self.log_dict(renamed_scores)

            if loss_info is not None:
                renamed_info = {'{}_{}'.format(crit_prefix, k): v for k, v in loss_info.items()} if crit_prefix else loss_info
                self.log_dict(renamed_info)
        
        if log_best_ever_scores:
            if not hasattr(self, 'best_Sum') or scores['Sum'] > self.best_Sum:
                self.best_Sum = scores['Sum']
                self.CIDEr_in_the_best = scores['CIDEr']
            
            if not hasattr(self, 'best_CIDEr') or scores['CIDEr'] > self.best_CIDEr:
                self.best_CIDEr = scores['CIDEr']
            
            self.log('best_Sum', self.best_Sum, prog_bar=True)
            self.log('best_CIDEr', self.best_CIDEr, prog_bar=True)
            # self.log('CIDEr_in_the_best', self.CIDEr_in_the_best, prog_bar=True)
        
        if getattr(self, 'eval_criterion', None) is not None:
            self.eval_criterion.reset_loss_recorder()

        return scores, detail_scores, preds
 
    def on_validation_epoch_start(self) -> None:
        self.prepare_auxiliary_info()
    
    def on_test_epoch_start(self) -> None:
        self.prepare_auxiliary_info()

    def on_validation_epoch_end(self) -> None:
        self.post_process_auxiliary_info()
    
    def on_test_epoch_end(self) -> None:
        self.post_process_auxiliary_info()

    def prepare_auxiliary_info(self) -> None:
        opt = self.get_opt()
        if opt['decoding_type'] == 'NARFormer' \
            and opt.get('teacher_path', '') \
            and not hasattr(self, 'teacher_model_wrapper'):
            # self.teacher_model_wrapper will be used in translate_step
            self.teacher_model_wrapper = Model.load_from_checkpoint(
                opt['teacher_path'], strict=True)
    
    def post_process_auxiliary_info(self) -> None:
        if hasattr(self, 'teacher_model_wrapper'):
            # this is to avoid saving self.teacher_model_wrapper to the checkpoint file
            del self.teacher_model_wrapper

    def get_info_corpus(self) -> Dict[str, Any]:
        opt = self.get_opt()
        if not hasattr(self, 'info_corpus'):
            self.info_corpus = pickle.load(open(opt['info_corpus'], 'rb'))
        return self.info_corpus

    def get_vocab(self) -> Dict[int, str]:
        info_corpus = self.get_info_corpus()
        return info_corpus['info']['itow']
    
    def get_references(self) -> Dict[str, List[dict]]:
        if not hasattr(self, 'references'):
            self.references = pickle.load(open(self.hparams.opt['reference'], 'rb'))
        return self.references
    
    def configure_optimizers(self):
        opt = self.get_opt()
        lr = opt.get('learning_rate', 5e-4)
        weight_decay = opt.get('weight_decay', 5e-4)

        if opt.get('filter_weight_decay', False):
            parameters = filter_weight_decay(
                model=self.captioner,
                weight_decay=weight_decay,
                filter_biases=opt.get('filter_biases', False),
                skip_list=(),
                skip_substr_list=opt.get('skip_substr_list', []),
            )
            optimizer = torch.optim.Adam(parameters, lr=lr)
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.captioner.parameters()), lr=lr, weight_decay=weight_decay)

        lr_scheduler_type = self.hparams.opt.get('lr_scheduler_type', 'linear')
        if lr_scheduler_type == 'linear':
            from torch.optim.lr_scheduler import StepLR
            lr_decay = self.hparams.opt.get('lr_decay', 0.9)
            lr_step_size = self.hparams.opt.get('lr_step_size', 1)

            lr_scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=lr_decay)
            other_info = {}
        elif lr_scheduler_type == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_steps, eta_min=self.hparams.opt.get('min_lr', 1e-6))
            other_info = {'interval': 'step'}
        elif lr_scheduler_type == 'linear_with_warmup':
            from misc.optim import get_linear_schedule_with_warmup
            if self.hparams.opt['learning_rate_warmup_ratio']:
                num_warmup_steps = int(self.hparams.opt['max_steps'] * self.hparams.opt['learning_rate_warmup_ratio'])
            else:
                num_warmup_steps = self.hparams.opt['learning_rate_warmup_steps']
            
            print('- warmup_steps:', num_warmup_steps)

            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer, 
                num_training_steps=self.hparams.opt['max_steps'], 
                num_warmup_steps=num_warmup_steps
            )
            other_info = {'interval': 'step'}
        else:
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            lr_decay = self.hparams.opt.get('lr_decay', 0.9)
            lr_monitor_mode = self.hparams.opt.get('lr_monitor_mode', 'max')
            lr_monitor_metric = self.hparams.opt.get('lr_monitor_metric', 'CIDEr')
            lr_monitor_patience = self.hparams.opt.get('lr_monitor_patience', 1)
            min_lr = self.hparams.opt.get('min_lr', 1e-6)

            lr_scheduler = ReduceLROnPlateau(
                optimizer, 
                mode=lr_monitor_mode, 
                factor=lr_decay, 
                patience=lr_monitor_patience,
                min_lr=min_lr
            )
            other_info = {'monitor': lr_monitor_metric, 'strict': True}
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'epoch',
                'frequency': 1,
                **other_info
            }
        }
    
    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None) # don't show the version number
        return items
    
    def get_keys_to_device(self, *arg, **kwargs):
        if isinstance(self.captioner, list):
            keys = set()
            for captioner in self.captioner:
                keys = keys | set(captioner.get_keys_to_device(*arg, **kwargs))
            return list(keys)
        else:
            return self.captioner.get_keys_to_device(*arg, **kwargs)

    def get_opt(self):
        return {**self.hparams.opt, **self.hparams.new_opt_used_to_override}
    
    def update_opt(self, new_opt):
        self.hparams.opt.update(new_opt)


class Model(ModelBase):
    def __init__(self, opt: Dict[str, Any], new_opt_used_to_override: Dict[str, Any] = {}, merge_opt: bool=False):
        if merge_opt:
            opt, new_opt_used_to_override = {**opt, **new_opt_used_to_override}, {}
        super().__init__(opt, new_opt_used_to_override)
        opt = self.get_opt()

        # for training
        self.criterion = get_criterion(opt, override_opt={'calculate_mAP': False})
        self.training_scales = opt.get('training_scales', {})

        # for evaluation
        self.eval_criterion = get_criterion(opt, skip_crit_list=['lang'], override_opt={'calculate_mAP': True})

    def training_step(self, batch, batch_idx):
        # self.current_epoch is automatically provided by `LightningModule`
        feedforward_results = self.captioner(batch, current_epoch=self.current_epoch)
        loss = self.criterion.get_loss({**feedforward_results, **batch})
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log('schedule_sampling_prob', feedforward_results['schedule_sampling_prob'], on_step=True, on_epoch=False, prog_bar=False)

        # optimizer = self.optimizers()
        # for pg in optimizer.param_groups:
        #     print(batch_idx, pg["lr"])

        # print(self.criterion.get_loss_info())
        return loss
    
    # def on_after_backward(self) -> None:
    #     for name, parms in self.captioner.named_parameters():	
    #         print('name:', name, 'grad_requirs:', parms.requires_grad, 'grad_value:', parms.grad.mean())
    
    def on_train_epoch_start(self):
        if self.current_epoch in self.training_scales.keys():
            scales = self.training_scales[self.current_epoch]
            self.criterion.set_scales(scales)
            print(f'- Epoch {self.current_epoch}, setting training scales to {scales}')

    def training_epoch_end(self, outputs) -> None:
        loss_info = self.criterion.get_loss_info()
        other_loss_info = {}
        for k in list(loss_info.keys()):
            # too many info for attribute prediction, we do not add them to the bar
            if ('F1-' in k and 'F1-30' not in k) or ('P-' in k):
                other_loss_info[k] = loss_info.pop(k)
        
        self.log_dict(loss_info, prog_bar=True)
        self.log_dict(other_loss_info, prog_bar=False)
        self.criterion.reset_loss_recorder()
    

class MultipleOptimizerModel(Model):
    def __init__(self, opt: Dict[str, Any], new_opt_used_to_override: Dict[str, Any] = {}, merge_opt: bool=False):
        super().__init__(opt, new_opt_used_to_override, merge_opt)
        self.automatic_optimization = False
    
    def configure_optimizers(self):
        opt = self.get_opt()
        lr = opt.get('learning_rate', 5e-4)
        weight_decay = opt.get('weight_decay', 5e-4)

        def add_weight_decay(param_list, weight_decay, lr):
            decay, no_decay = [], []
            for param in param_list:
                if not param.requires_grad:
                    continue  # frozen weights
                if param.dim() == 1:
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [
                {'params': decay, 'weight_decay': weight_decay, 'lr': lr},
                {'params': no_decay, 'weight_decay': 0., 'lr': lr},
            ]
        
        # normal optimizer, same learning rate for all parameters
        if opt.get('filter_weight_decay', False) and opt.get('filter_biases', False):
            print('- We do not apply weight decay to gains and biases')
            params_group = add_weight_decay(list(self.captioner.parameters()), weight_decay, lr)
            optimizer = torch.optim.Adam(params_group)
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.captioner.parameters()), lr=lr, weight_decay=weight_decay)

        # set a low learning rate for the encoder and the concept detector
        names = ['encoder', 'predictor.nets.0']
        if opt['decoding_type'] == 'NARFormer':
            names.append('predictor.nets.1')

        low_lr = opt.get('low_learning_rate', 5e-5)

        low_params = []
        params = []
        for n, p in self.captioner.named_parameters():
            if p.requires_grad:
                flag = sum([1 for specified_name in names if specified_name in n])
                if flag:
                    low_params.append(p)
                else:
                    params.append(p)
        
        if opt.get('filter_weight_decay', False) and opt.get('filter_biases', False):
            print('- We do not apply weight decay to gains and biases')
            params_group1 = add_weight_decay(low_params, weight_decay, low_lr)
            params_group2 = add_weight_decay(params, weight_decay, lr)
            optimizer_lowlr = torch.optim.Adam(params_group1 + params_group2)
        else:
            optimizer_lowlr = torch.optim.Adam([
                {"params": low_params, "lr": low_lr},
                {"params": params, "lr": lr}
            ], weight_decay=weight_decay)
        
        from torch.optim.lr_scheduler import StepLR
        lr_decay = self.hparams.opt.get('lr_decay', 0.9)
        lr_step_size = self.hparams.opt.get('lr_step_size', 1)
        lr_scheduler = StepLR(optimizer, step_size=lr_step_size, gamma=lr_decay)
        lr_scheduler2 = StepLR(optimizer_lowlr, step_size=lr_step_size, gamma=lr_decay)
 
        return [optimizer, optimizer_lowlr], [lr_scheduler, lr_scheduler2]
    
    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)

        optim1, optim2 = self.optimizers()
        optim = optim2 if self.current_epoch >= self.hparams.opt['lowlr_start_epoch'] else optim1
        
        optim.zero_grad()
        loss.backward()
        optim.step()    
    
    def training_epoch_end(self, outputs) -> None:
        super().training_epoch_end(outputs)
        sch1, sch2 = self.lr_schedulers()
        sch1.step()
        sch2.step()
        # _, optimizer = self.optimizers()
        # print()
        # for pg in optimizer.param_groups:
        #    print(pg["lr"])


class InterplayModel(Model):
    def __init__(self, opt: Dict[str, Any], new_opt_used_to_override: Dict[str, Any] = {}, merge_opt: bool=False):
        super().__init__(opt, new_opt_used_to_override, merge_opt)
        newest_opt = self.get_opt()
        self.teacher_captioner = get_framework(newest_opt)
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        student_results = self.captioner(batch, current_epoch=self.current_epoch)
        captioning_loss = self.criterion.get_loss({**student_results, **batch})

        # Knowledge distillation
        with torch.no_grad():
            teacher_results = self.teacher_captioner(batch, current_epoch=self.current_epoch)

        distillation_loss = ((student_results['logits'] - teacher_results['logits']) ** 2).mean()
        
        loss = captioning_loss + self.get_opt().get('distillation_weight', 0.01) * distillation_loss

        optim = self.optimizers()
        optim.zero_grad()
        loss.backward()
        optim.step()

        # EMA update
        with torch.no_grad():
            ema_weight = self.get_opt().get('ema_weight', 0.999)
            params_s = list(self.captioner.parameters())
            params_t = list(self.teacher_captioner.parameters())
            torch._foreach_mul_(params_t, ema_weight)
            w = torch._foreach_mul(params_s, 1 - ema_weight)
            torch._foreach_add_(params_t, w)

        self.log('captioning_loss', captioning_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log('distillation_loss', distillation_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log('total_loss', loss, on_step=True, on_epoch=True, prog_bar=False)
        return loss

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.swap_captioners()
    
    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()
        self.swap_captioners()
    
    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        self.swap_captioners()
    
    def on_test_epoch_end(self) -> None:
        super().on_test_epoch_end()
        self.swap_captioners()
    
    def swap_captioners(self):
        if self.get_opt().get('eval_model', 'teacher') == 'teacher':
            self.teacher_captioner, self.captioner = self.captioner, self.teacher_captioner

    def training_epoch_end(self, outputs) -> None:
        super().training_epoch_end(outputs)
        sch = self.lr_schedulers()
        sch.step()
        #optimizer = self.optimizers()
        #for pg in optimizer.param_groups:
        #    print( pg["lr"])


class ModelEnsemble(ModelBase):
    def __init__(self, 
            checkpoint_paths: List[str], 
            new_opt_used_to_override: Dict[str, Any] = {},
            map_location: Optional[torch.device] = None,
            strict: bool = True,
            WRAPPER = Model,
        ):
        '''
            args:
                checkpoint_paths:           a list of pre-trained models' path
                new_opt_used_to_override:   some new arguments, e.g., using a new beam size during inference 
                                            by passing `new_opt_used_to_override` = {'beam_size': 1}
        '''
        assert isinstance(checkpoint_paths, list)
        assert len(checkpoint_paths) >= 1

        all_captioners = []
        opt = None

        modality_of_all_checkpoints = []
        full_modality = ''

        for checkpoint_path in checkpoint_paths:
            model = WRAPPER.load_from_checkpoint(checkpoint_path, map_location=map_location, strict=strict)
            all_captioners.append(model.captioner)

            modality_of_all_checkpoints.append(model.hparams.opt['modality'])
            full_modality += model.hparams.opt['modality']

            #assert model.hparams.opt['decoding_type'] != 'NARFormer', \
            #    'we do not support ensembling non-autoregressive models'

            if opt is None:
                opt = model.hparams.opt
            else:
                # we need to check if two checkpoints' modalities and features are different
                new_opt = model.hparams.opt
                for char in new_opt['modality']:
                    if char in opt['modality']:
                        # we hope that the same modality uses the same features
                        new_feats, feats = new_opt['feats_%s'%char], opt['feats_%s'%char]
                        assert len(new_feats) == len(feats), f"{new_feats}, {feats}"
                        for p1, p2 in zip(new_feats, feats):
                            assert p1 == p2, f"{p1}, {p2}"
                    else:
                        opt['feats_%s'%char] = new_opt['feats_%s'%char]
        
        if len(set(modality_of_all_checkpoints)) == 1:
            # all checkpoints use the same modalities and feats
            self.need_to_split_feats = False
        else:
            opt['modality'] = ''
            for char in list(set(full_modality)):
                opt['modality'] += char
            self.need_to_split_feats = True
            self.modality_of_all_checkpoints = modality_of_all_checkpoints

        super().__init__(opt, new_opt_used_to_override)
        del self.captioner
        self.captioner = all_captioners
    
    def preprocess_batch_before_translate_step(self, batch):
        if self.need_to_split_feats:
            all_feats = []
            
            for modality in self.modality_of_all_checkpoints:
                this_feats = []
                for char in modality:
                    index_of_this_char = self.hparams.opt['modality'].index(char)
                    this_feats.append(batch['feats'][index_of_this_char])
                all_feats.append(this_feats)
            
            batch['feats'] = all_feats

        return batch

    def training_step(self, batch, batch_idx):
        raise NotImplementedError('not supporting the training of multiple models yet!')
        
    def training_epoch_end(self, outputs) -> None:
        raise NotImplementedError('not supporting the training of multiple models yet!')
    
    def train(self) -> None:
        for model in self.captioner:
            model.train()
    
    def eval(self) -> None:
        for model in self.captioner:
            model.eval()
    
    def to(self, device) -> None:
        for model in self.captioner:
            model.to(device)
    
    def parameters(self):
        import itertools
        return itertools.chain(*[model.parameters() for model in self.captioner])

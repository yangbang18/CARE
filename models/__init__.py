from .Wrapper import Model, ModelEnsemble, InterplayModel, MultipleOptimizerModel
import torch
from config import Constants
import os


def modify_opt_if_necessary(args, model):
    opt = model.get_opt()

    if getattr(args, 'retrieval_datasets', []):
        assert opt['feats_r']
        assert 'CLIP_ViT-B-32' in opt['feats_r'], opt['feats_r']
        assert 'unique' in opt['feats_r'], opt['feats_r']
        if len(args.retrieval_datasets) == 1 and args.retrieval_datasets[0] == 'MSRVTT':
            opt['feats_r'] = os.path.join(os.path.dirname(opt['feats_r']), 'CLIP_ViT-B-32_unique.hdf5')
        else:
            opt['feats_r'] = os.path.join(os.path.dirname(opt['feats_r']), 'CLIP_ViT-B-32_{}_unique.hdf5'.format('-'.join(args.retrieval_datasets)))
    if getattr(args, 'retrieval_db_ratio', 100) < 100:
        assert opt['feats_r'] or opt['feats_t']
        if opt['feats_r']:
            if isinstance(opt['feats_r'], (list, tuple)):
                assert len(opt['feats_r']) == 1
                opt['feats_r'] = opt['feats_r'][0]
            opt['feats_r'] = opt['feats_r'].replace('.hdf5', '_ratio%.1f.hdf5' % args.retrieval_db_ratio)
            print('- Modify feats_r to', opt['feats_r'])
        if opt['feats_t']:
            opt['feats_t'] = opt['feats_t'].replace('.hdf5', '_ratio%.1f.hdf5' % args.retrieval_db_ratio)
            print('- Modify feats_t to', opt['feats_t'])
    
    model.hparams.opt = opt
    model.hparams.new_opt_used_to_override = {}
    return model


def load_model_from_arguments(args, ignore_empty_attributes=[], replace_paths=True, pluggin_func=modify_opt_if_necessary):
    if getattr(args, 'no_cuda', False) \
        or getattr(args, 'gpus', 1) == 0 \
            or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    
    ensemble_flag = False
    if hasattr(args, 'checkpoint_path'):
        assert type(args.checkpoint_path) is str
        checkpoint_path = args.checkpoint_path
    elif hasattr(args, 'checkpoint_paths'):
        assert isinstance(args.checkpoint_paths, (list, tuple))
        checkpoint_path = args.checkpoint_paths
        
        if len(checkpoint_path) > 1:
            ensemble_flag = True
        else:
            checkpoint_path = checkpoint_path[0]
    else:
        raise AttributeError('Neither `checkpoint_path` or `checkpoint_paths` is found in the given arguments')

    if hasattr(args, 'wrapper'):
        WRAPPER = eval(args.wrapper)
    else:
        WRAPPER = Model

    strict = False
    if getattr(args, 'load_strictly', False) or getattr(args, 'strict', False):
        strict = True
    elif hasattr(args, 'with_backbones') and not args.with_backbones:
        strict = True
        del args.with_backbones
    
    base_data_path = getattr(args, 'base_data_path', Constants.base_data_path)

    for attr in ignore_empty_attributes:
        if hasattr(args, attr) and not getattr(args, attr):
            delattr(args, attr)    
    
    model = load_model(
        checkpoint_path=checkpoint_path, 
        new_opt_used_to_override=vars(args), 
        device=device, 
        strict=strict, 
        WRAPPER=WRAPPER, 
        replace_paths=replace_paths,
        base_data_path=base_data_path,
        ensemble_flag=ensemble_flag,
    )

    if pluggin_func is not None:
        model = pluggin_func(args, model)

    return model


def load_model(
        checkpoint_path, 
        new_opt_used_to_override={}, 
        device=torch.device('cpu'),  
        strict=True,
        WRAPPER=Model,
        replace_paths=True,
        base_data_path=None,
        ensemble_flag=None,
    ):
    if ensemble_flag is None:
        ensemble_flag = isinstance(checkpoint_path, (list, tuple))

    if ensemble_flag:
        model = ModelEnsemble(
            checkpoint_path, 
            new_opt_used_to_override=new_opt_used_to_override,
            map_location=device,
            strict=strict,
            WRAPPER=WRAPPER,
        )
    else:
        model = WRAPPER.load_from_checkpoint(
            checkpoint_path, 
            new_opt_used_to_override=new_opt_used_to_override,
            map_location=device,
            strict=strict
        )

    if replace_paths:
        # this should be run if you want to evaluate pre-trained models released by others
        # because the evluation code (e.g., translate.py) will call model.get_opt() to define a dataloader
        # so paths to load features / corpus / references should be correct
        opt = model.get_opt()
        
        ori_base_data_path = os.path.dirname(opt['info_corpus'])
        assert os.path.basename(ori_base_data_path) == opt['dataset']
        ori_base_data_path = os.path.dirname(ori_base_data_path)

        now_base_data_path = base_data_path if base_data_path is not None else Constants.base_data_path

        def _replace(item, src, trg):
            if isinstance(item, (list, tuple)):
                return [_replace(_, src, trg) for _ in item]
            
            assert type(item) is str
            return item.replace(src, trg)

        for key in ['feats_a', 'feats_m', 'feats_i', 'feats_o', 'feats_t', 'feats_r'] \
                + ['reference', 'info_corpus']:
            if key not in opt:
                continue
            opt[key] = _replace(opt[key], ori_base_data_path, now_base_data_path)
        
        model.hparams.opt = opt
        model.hparams.new_opt_used_to_override = {}
    
    model.eval()
    model.to(device)
    return model


def manually_load_pretrained_teacher_model(opt):
    WRAPPER = eval(opt.get('wrapper', 'Model'))
    model = WRAPPER(opt)
    
    checkpoint = torch.load(opt['teacher_path'], map_location='cpu')
    teacher_state_dict = checkpoint['state_dict']
    student_state_dict = model.state_dict()

    set_teacher = set(teacher_state_dict.keys())
    set_student = set(student_state_dict.keys())
    if len(list(set_student - set_teacher)):
        print('- Unexpected Keys:', list(set_student - set_teacher))
    if len(list(set_teacher - set_student)):
        print('- Extra Keys in the Checkpoint:', list(set_teacher - set_student))

    from .Translator import get_vocab_mapping
    vocab_mapping = get_vocab_mapping(
        opt=model.get_opt(), 
        teacher_opt=checkpoint['hyper_parameters']['opt']
    )

    for k, v in student_state_dict.items():
        if k in teacher_state_dict:
            teacher_v = teacher_state_dict[k]
            if teacher_v.shape == v.shape:
                student_state_dict[k] = teacher_v
            else:
                print(f'- Incompatible Shape of `{k}`: Student {v.shape}; Teacher {teacher_v.shape}')
                if 'word_embeddings' in k or 'tgt_word_prj' in k:
                    if vocab_mapping is not None:
                        print('- Applying Vocab Mapping')
                        student_state_dict[k] = teacher_v[vocab_mapping]

    model.load_state_dict(student_state_dict)

    return model



# def postprocessing_pretrained_model(model):
#     opt = model.get_opt()
#     if opt['decoding_type'] == 'NARFormer' and opt.get('load_teacher_weights', False):
#         from .Translator import get_vocab_mapping

#         assert opt['distilled_info_corpus_name']

#         model.prepare_auxiliary_info()
#         teacher_opt = model.teacher_model_wrapper.get_opt()

#         opt['info_corpus'] = os.path.join(os.path.dirname(opt['info_corpus']), opt.pop('distilled_info_corpus_name'))
#         vocab_mapping = get_vocab_mapping(opt, teacher_opt)
#         if vocab_mapping is None:
#             return model

#         ori_vocab_size, opt['vocab_size'] = opt['vocab_size'], len(vocab_mapping)
#         print(f'- Chaning vocab size from {ori_vocab_size} to', opt['vocab_size'])

#         print('- Resizing word embeddings and the classification head')
#         model.captioner.cls_head.resize_word_embeddings(vocab_mapping)
#         model.captioner.decoder.resize_word_embeddings(vocab_mapping)
#         model.update_opt(opt)
#         model.post_process_auxiliary_info()
#         print('- Done!')
    
#     return model

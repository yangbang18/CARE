import argparse
from config import Constants
import os
import pickle
from misc.utils import (
    load_yaml, 
    check_whether_to_load_weights,
)
from models.Predictor import (
    add_predictor_specific_args,
    check_predictor_args,
)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='MSRVTT', choices=['MSVD', 'MSRVTT', 'VATEX'])
    parser.add_argument('-m', '--modality', type=str, default='mi')
    parser.add_argument('-scope', '--scope', type=str, default='')
    parser.add_argument('-method', '--method', type=str, default='', help='method to use, defined in config/methods.yaml')
    parser.add_argument('-task', '--task', type=str, default='', help='task to use, defined in config/tasks.yaml')
    parser.add_argument('-feats', '--feats', type=str, default='', help='features to use, defined in config/feats.yaml')
    parser.add_argument('-arch', '--arch', type=str, default='base', help='architecture to use, defined in config/feats.yaml')
    parser.add_argument('-setup', '--setup', type=str, default='naive', help='training setup to use, defined in config/setups.yaml')

    parser.add_argument('--wrapper', type=str, default='Model', choices=['Model', 'InterplayModel'])

    parser.add_argument('-pte', '--pretrain_epochs', type=int, default=10)

    # the following arguments may be overrided by the config in specified method
    parser.add_argument('--encoder', type=str, default='Embedder', help='encoder to use')
    parser.add_argument('--decoder', type=str, default='TransformerDecoder', help='decoder to use')
    parser.add_argument('--pointer', type=str, help='Pointer netowrk to use (retrieval-based method)')
    parser.add_argument('--cls_head', type=str, default='NaiveHead', help='classification head to use')
    parser.add_argument('--decoding_type', type=str, default='ARFormer', help='ARFormer | NARFormer')
    parser.add_argument('--fusion', type=str, default='temporal_concat', help='temporal_concat | addition')

    parser.add_argument('--copy_scale', type=float, default=1.0)
    parser.add_argument('--exclude_eos', action='store_true')
    parser.add_argument('--has_retrieval_embs', action='store_true')
    parser.add_argument('--has_retrieval_rnn', action='store_true')

    parser = add_predictor_specific_args(parser)
    parser.add_argument('--num_sanity_val_steps', type=int, default=0)

    model = parser.add_argument_group(title='Common Model Settings')
    model.add_argument('--dim_hidden', type=int, default=512, help='size of the hidden layer')
    model.add_argument('--encoder_dropout_prob', type=float, default=0.5, help='strength of dropout in the encoder')
    model.add_argument('--hidden_dropout_prob', type=float, default=0.5, help='strength of dropout in the decoder')
    model.add_argument('-wc', '--with_category', default=False, action='store_true',
                        help='specified for the MSRVTT dataset, use category tags or not')
    model.add_argument('--num_category', type=int, default=20)
    model.add_argument('--use_category_embs', default=False, action='store_true')
    model.add_argument('--dim_category', type=int, default=300)

    model.add_argument('--pretrained_embs_path', type=str, default='', 
                        help='path to load pretrained word embs, which will be fixed if specified; '
                        'default to empty string, i.e., the model uses trainable word embs of dimension `dim_hidden`')
    model.add_argument('--load_model_weights_from', type=str, default='',
                        help='if specified, initializing the model with specific checkpoint file (not strict)')
    model.add_argument('--load_strictly', default=False, action='store_true')
    model.add_argument('--freeze_parameters_except', type=str, default=[], nargs='+',
                        help='when `load_model_weights_from` is True, specified parameters will not be frozen during training; '
                        'if not specified (default), all paramters are trainable')
    model.add_argument('--with_backbones', type=str, nargs='+', default=[])

    model_tf = parser.add_argument_group(title='Transformer Model Settings')
    model_tf.add_argument('--transformer_pre_ln', default=False, action='store_true', 
        help='refer to `On Layer Normalization in the Transformer Architecture` http://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf')
    model_tf.add_argument('--trainable_pe', default=False, action='store_true', help='use fixed (default) or trainable positional embs')
    model_tf.add_argument('--mha_exclude_bias', default=False, action='store_true')
    model_tf.add_argument('-nel', '--num_hidden_layers_encoder', type=int, default=1)
    model_tf.add_argument('-ndl', '--num_hidden_layers_decoder', type=int, default=1)
    
    model_tf.add_argument('-ntl', '--num_hidden_layers_text', type=int, default=1)
    model_tf.add_argument('--crosslayer_no_ffn', default=False, action='store_true')

    model_tf.add_argument('--num_attention_heads', type=int, default=8)
    model_tf.add_argument('--intermediate_size', type=int, default=2048)
    model_tf.add_argument('--hidden_act', type=str, default='relu')
    model_tf.add_argument('--attention_probs_dropout_prob', type=float, default=0.1)
    model_tf.add_argument('--layer_norm_eps', type=float, default=1e-12)
    model_tf.add_argument('--watch', type=int, default=0)
    model_tf.add_argument('--pos_attention', default=False, action='store_true')
    model_tf.add_argument('--enhance_input', type=int, default=2, 
                        help='for NA decoding, 0: without R | 1: re-sampling(R)) | 2: meanpooling(R), default to 2',
                        choices=[0, 1, 2])

    model_tf.add_argument('-RPE', '--RPE', default=False, action='store_true')
    model_tf.add_argument('-keep', '--RPE_keep_abs_pos', default=False, action='store_true')
    model_tf.add_argument('-mrp', '--max_relative_position', type=int, default=30)

    model_rnn = parser.add_argument_group(title='RNN Model Settings')
    model_rnn.add_argument('--rnn_type', default='lstm', type=str, help='the basic unit of RNN based decoders', choices=['lstm', 'gru'])
    model_rnn.add_argument('--with_multileval_attention', default=False, action='store_true', 
                        help='also known as multimodal attention or attentional attention')
    model_rnn.add_argument('--feats_share_weights', default=False, action='store_true', 
                        help='in temporal attention, share the weights of different features or not')


    training = parser.add_argument_group(title='Common Training Settings')
    training.add_argument('-gpus', '--gpus', default=1, type=int, 
                        help='the number of gpus to use, only support 0 (cpu) and 1 now', choices=[0, 1])
    training.add_argument('-devices', '--devices', type=str, default='')
    training.add_argument('-seed', '--seed', default=0, type=int, help='for reproducibility')
    training.add_argument('-e', '--epochs', type=int, default=50, help='number of epochs')
    training.add_argument('-b', '--batch_size', type=int, default=64, help='minibatch size')
    training.add_argument('--max_steps', type=int ,default=None, 
                        help='training will stop if `max_steps` or `epochs` have reached (earliest), default to None')
    
    training.add_argument('--skip_substr_list', nargs='+', type=str, default=[])

    training_rnn = parser.add_argument_group(title='RNN Training Settings')
    # scheduled sampling: https://arxiv.org/pdf/1506.03099.pdf
    training_rnn.add_argument('--scheduled_sampling_start', type=int, default=-1)
    training_rnn.add_argument('--scheduled_sampling_increase_every', type=int, default=5)
    training_rnn.add_argument('--scheduled_sampling_increase_prob', type=float, default=0.05)
    training_rnn.add_argument('--scheduled_sampling_max_prob', type=float, default=0.25)

    training_na = parser.add_argument_group(title='Non-Autoregressive Model Training Settings')
    training_na.add_argument('--with_teacher_during_training', default=False, action='store_true')
    training_na.add_argument('--teacher_path', type=str, default='', help='path for the AR-B model')
    training_na.add_argument('--teacher_scope', type=str, default='', help='scope for the AR-B model')
    training_na.add_argument('--beta', nargs='+', type=float, default=[0, 1],
                        help='len=2, [lowest masking ratio, highest masking ratio]')
    training_na.add_argument('--visual_word_generation', default=False, action='store_true')
    training_na.add_argument('--demand', nargs='+', type=str, default=['VERB', 'NOUN'], 
                        help='pos_tag we want to focus on when training with visual word generation')
    training_na.add_argument('-nvw', '--nv_weights', nargs='+', type=float, default=[0.8, 1.0],
                        help='len=2, weights of visual word generation and caption generation (or mlm)')
    training_na.add_argument('--load_teacher_weights', default=False, action='store_true',
                        help='specified for NA-based models, initialize randomly or inherit from the teacher (AR-B)')


    optim_scheduler = parser.add_argument_group(title='Optimizer & LR Scheduler Settings')
    optim_scheduler.add_argument('--learning_rate', default=5e-4, type=float, help='the initial larning rate')
    optim_scheduler.add_argument('--learning_rate_warmup_steps', default=1000, type=int, help='the number of steps to reach the peak')
    optim_scheduler.add_argument('--learning_rate_warmup_ratio', default=0.0, type=float, help='the ratio of steps to reach the peak')
    optim_scheduler.add_argument('--weight_decay', type=float, default=0.001, help='strength of weight regularization')
    optim_scheduler.add_argument('--filter_weight_decay', default=False, action='store_true', 
                        help='do not apply weight_decay on specific parametes')
    optim_scheduler.add_argument('--filter_biases', default=False, action='store_true',
                        help='if True, not applying weight decay on biases')

    optim_scheduler.add_argument('--gradient_clip_val', default=0.0, type=float, help='gradient clipping value')
    optim_scheduler.add_argument('--lr_scheduler_type', default='linear', type=str, 
                        help='`linear` (default): StepLR | otherwise: ReduceLROnPlateau', choices=['linear', 'plateau', 'cosine', 'linear_with_warmup'])
    # if `lr_scheduler_type` == 'linear'
    optim_scheduler.add_argument('--lr_decay', default=0.9, type=float, help='the decay rate of learning rate per epoch')
    optim_scheduler.add_argument('--lr_step_size', default=1, type=int, help='period of learning rate decay')
    # otherwise
    optim_scheduler.add_argument('--lr_monitor_mode', default='max', type=str, 
                        help='max (default): higher the metric, better the performance | min: just the opposite',
                        choices=['min', 'max'])
    optim_scheduler.add_argument('--lr_monitor_metric', default='CIDEr', type=str, help='specify the metric for lr adjustment')
    optim_scheduler.add_argument('--lr_monitor_patience', default=1, type=int, help='number of epochs with no improvement after which lr will be reduced')
    optim_scheduler.add_argument('--min_lr', default=1e-6, type=float, help='the minimum learning rate')
    
    optim_scheduler.add_argument('--low_learning_rate', type=float, default=5e-5)
    optim_scheduler.add_argument('--lowlr_start_epoch', type=int, default=10)


    evaluation = parser.add_argument_group(title='Common Evaluation Settings')
    evaluation.add_argument('--check_val_every_n_epoch', type=int, default=1, 
                        help='check on the validation set every n train epochs, default to 1')
    evaluation.add_argument('--metric_sum', nargs='+', type=int, default=[1, 1, 1, 1],
                        help='which metrics to calculate `Sum`, default to [1, 1, 1, 1], '
                        'i.e., `Sum` = `Bleu_4` + `METEOR` + `ROUGE_L` + `CIDEr`')
    evaluation.add_argument('--save_csv', default=False, action='store_true',
                        help='save test results to csv file')
    evaluation.add_argument('--VATEX_I3D_preds_json', type=str, default='', help='use to complete predictions for those missing videos in VATEX')

    evaluation_ar = parser.add_argument_group(title='Autoregressive Model Evaluation Settings')
    evaluation_ar.add_argument('-bs', '--beam_size', type=int, default=5,
                        help='specified for AR decoding, the number of candidates')
    evaluation_ar.add_argument('-ba', '--beam_alpha', type=float, default=1.0,
                        help='the preference of the model towards the average sentence length, '
                        'the larger `beam_alpha` is, the longer is the average sentence length')
    
    evaluation_na = parser.add_argument_group(title='Non-Autoregressive Model Evaluation Settings')
    evaluation_na.add_argument('--paradigm', type=str, default='mp', 
                        help='mp: MaskPredict | l2r: Left2Right | ef: EasyFirst')
    evaluation_na.add_argument('-lbs', '--length_beam_size', type=int, default=6,
                        help='specified for NA decoding, the number of length candidates')
    evaluation_na.add_argument('--iterations', type=int, default=5,
                        help='the number of iterations for the MP algorithm')
    evaluation_na.add_argument('--q', type=int, default=1,
                        help='the number of tokens to update for L2R & EF algorithms')
    evaluation_na.add_argument('--q_iterations', type=int, default=1,
                        help='the number of iterations for L2R & EF algorithms')
    evaluation_na.add_argument('--use_ct', default=False, action='store_true', 
                        help='use coarse-grained templates or not, only for methods with visual word generation')


    checkpoint = parser.add_argument_group(title='Checkpoint Settings')
    checkpoint.add_argument('--monitor_metric', type=str, default='CIDEr',
                            help='which metric to monitor for checkpoint saving: Bleu_4 | METEOR | ROUGE_L | CIDEr (default) | Sum',
                            choices=['Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'Sum'])
    checkpoint.add_argument('--monitor_mode', type=str, default='max',
                            help='max: the higher the `monitor_metric` the better the performance | min: just the opposite',
                            choices=['min', 'max'])    
    checkpoint.add_argument('--save_topk_models', type=int, default=1,
                            help='checkpoints with top-k performance will be saved, default to 1')
    checkpoint.add_argument('-sse', '--start_saving_epoch', type=int, default=0)


    dataloader = parser.add_argument_group(title='Dataloader Settings')
    dataloader.add_argument('--base_data_path', type=str, default='', help='if not specified, Constants.base_data_path will be used by default')
    dataloader.add_argument('--max_len', type=int, default=30, help='max length of captions')
    dataloader.add_argument('--n_frames', type=int, default=28, help='the number of frames to represent a whole video')
    dataloader.add_argument('--n_caps_per_video', type=int, default=0, 
                            help='the number of captions per video to constitute the training set, '
                            'default to 0 (i.e., loading all captions for a video)')
    dataloader.add_argument('--random_type', type=str, default='equally_sampling', 
                            help='sampling strategy during training: segment_random | all_random | equally_sampling (default)',
                            choices=['segment_random', 'all_random', 'equally_sampling'])
    dataloader.add_argument('--load_feats_type', type=int, default=1, 
                            help='load feats from the same frame_ids (0) '
                            'or different frame_ids (1, default), '
                            'or directly load all feats without sampling (2)', 
                            choices=[0, 1, 2])
    dataloader.add_argument('--num_workers', type=int, default=1, help='num_workers for dataloader, speed up training')

    # modality information
    dataloader.add_argument('--dim_a', type=int, default=1, help='feature dimension of the audio modality')
    dataloader.add_argument('--dim_m', type=int, default=2048, help='feature dimension of the motion modality')
    dataloader.add_argument('--dim_i', type=int, default=2048, help='feature dimension of the image modality')
    dataloader.add_argument('--dim_o', type=int, default=1, help='feature dimension of the object modality')
    dataloader.add_argument('--dim_t', type=int, default=1)
    dataloader.add_argument('--dim_r', type=int, default=1)
    dataloader.add_argument('--feats_a_name', nargs='+', type=str, default=[])
    dataloader.add_argument('--feats_m_name', nargs='+', type=str, default=['motion_resnext101_kinetics_duration16_overlap8.hdf5'])
    dataloader.add_argument('--feats_i_name', nargs='+', type=str, default=['image_resnet101_imagenet_fps_max60.hdf5'])
    dataloader.add_argument('--feats_o_name', nargs='+', type=str, default=[])
    dataloader.add_argument('--feats_t_name', nargs='+', type=str, default=[])
    dataloader.add_argument('--feats_r_name', nargs='+', type=str, default=[])
    dataloader.add_argument('--itoc_path', type=str, default='')
    # corpus information
    dataloader.add_argument('--info_corpus_name', type=str, default='info_corpus.pkl')
    dataloader.add_argument('-dicn', '--distilled_info_corpus_name', type=str, 
                            help='sequence-level knowledge distillation for non-autoregressive video captioning')
    dataloader.add_argument('--reference_name', type=str, default='refs.pkl')
    
    multitask = parser.add_argument_group(title='Multi-Task Settings')
    multitask.add_argument('--crits', nargs='+', type=str, default=['lang'], 
                            help='which training objectives to use')
    multitask.add_argument('--language_generation_scale', type=float, default=1.0, help='weight for the language generation task (`lang`)')

    multitask.add_argument('--label_smoothing', default=0., type=float,
                           help='label smoothing alpha, default: 0.0, no smoothing at all')
    
    mt = parser.add_argument_group(title='Mean Teacher Settings')
    mt.add_argument('--distillation_weight', type=float, default=0.01)
    mt.add_argument('--ema_weight', type=float, default=0.999)
    mt.add_argument('--eval_model', type=str, default='teacher', choices=['teacher', 'student'])
    args = parser.parse_args()
    return args


def load_yaml_to_update_args(args):
    load_yaml(args, args.method, yaml_path='./config/methods.yaml')
    check_whether_to_load_weights(args)
    load_yaml(args, args.task, yaml_path='./config/tasks.yaml', modify_scope=True, name_to_path=True)
    load_yaml(args, args.setup, yaml_path='./config/setups.yaml')
    load_yaml(args, args.feats, yaml_path='./config/feats.yaml')
    load_yaml(args, args.arch, yaml_path='./config/archs.yaml')
    

def get_dir(args, key, mid_path='', value=None):
    base_path = args.base_data_path if args.base_data_path else Constants.base_data_path
    
    if value is None:
        value = getattr(args, key, '')
    
    if not value:
        return ''

    if isinstance(value, list):
        return [get_dir(args, key, mid_path, value=v) for v in value]
    else:
        return os.path.join(base_path, args.dataset, mid_path, value)


def where_to_save_model(args):
    return os.path.join(
        Constants.base_checkpoint_path,
        args.dataset,
        args.method,
        args.task,
        args.scope
    )


def get_opt():
    args = parse_opt()
    load_yaml_to_update_args(args)

    if not args.task:
        assert args.scope, "Please add the argument \'--scope $folder_name_to_save_models\' or \'--task $task_name\'"
    
    if args.dataset in ['MSVD', 'VATEX']:
        if args.with_category:
            print(f"- Category information is not available in the {args.dataset} dataset, set `with_category` to False")
            args.with_category = False

    # log files and the best model will be saved at 'checkpoint_path'
    args.checkpoint_path = where_to_save_model(args)
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    
    # check teacher_path of NACF
    if args.decoding_type == 'NARFormer' and args.with_teacher_during_training:
        if not args.teacher_path:
            assert args.method == 'NACF', f'method should be NACF, but receive `{args.method}`'
            assert 'NACF' in args.checkpoint_path, args.checkpoint_path
            args.teacher_path = os.path.join(
                args.checkpoint_path.replace('NACF', 'ARB'),
                'best.ckpt'
            )
        assert os.path.exists(args.teacher_path), args.teacher_path
        
        if args.load_teacher_weights:
            args.load_model_weights_from = args.teacher_path
            args.load_strictly = False


    # get full paths to load features / corpora
    for key in ['feats_a_name', 'feats_m_name', 'feats_i_name', 'feats_o_name', 'feats_t_name', 'feats_r_name'] \
        + ['reference_name', 'info_corpus_name']:

        mid_path = ''
        if key == 'feats_r_name':
            mid_path = 'retrieval'
        elif 'feats' in key:
            mid_path = 'feats'

        if key == 'info_corpus_name' and args.distilled_info_corpus_name:
            assert args.decoding_type == 'NARFormer'
            new_key = 'distilled_info_corpus_name'
            setattr(args, key[:-5], get_dir(args, new_key, mid_path))
            delattr(args, key)
            delattr(args, new_key)
        else:
            setattr(args, key[:-5], get_dir(args, key, mid_path))
            delattr(args, key)
        
        print(key[:-5], getattr(args, key[:-5]))
    
    args.vocab_size = len(pickle.load(open(args.info_corpus, 'rb'))['info']['itow'].keys())
    check_predictor_args(args)
    opt = vars(args)
    print(opt)

    return opt

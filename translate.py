import sys
import os
import torch
import json

from models import load_model_from_arguments
from dataloader import get_loader
from pytorch_lightning import Trainer
import argparse
from tqdm import tqdm

from misc.utils import to_device, save_dict_to_csv
import time


def run_eval(
        args, 
        model, 
        loader: torch.utils.data.DataLoader, 
        device: torch.device,
        return_details: bool = False,
        only_return_pred_captions: bool = False,
    ):
    model.eval()
    model.to(device)
    
    vocab = model.get_vocab()

    latency = getattr(args, 'latency', False)
    save_latency = getattr(args, 'save_latency', False)
    total_time = 0

    all_step_outputs = []
    for batch in tqdm(loader):
        with torch.no_grad():
            for k in model.get_keys_to_device():
                if k in batch:
                    batch[k] = to_device(batch[k], device)

            if latency:
                start_time = time.time()

            step_outputs = model.translate_step(
                batch=batch,
                vocab=vocab,
                verbose=getattr(args, 'verbose', False),
                inference_latency=latency,
            )

            if latency:
                total_time += (time.time() - start_time)

        all_step_outputs.append(step_outputs)
    
    if latency:
        print(f'- # samples: {len(loader)}')
        print(f'- Total inference time: {total_time}')
        print(f'- Average latency: {total_time / len(loader)}')
        if save_latency:
            opt = model.get_opt()
            with open('latency.txt', 'a') as f:
                f.write('\t'.join([opt['method'], opt['task'], str(total_time), str(len(loader)), str(total_time/len(loader))]) + '\n')

        return total_time / len(loader)
    
    save_csv_path = None
    if getattr(args, 'csv_path', None):
        save_csv_path = args.csv_path

    scores, detail_scores, pred_captions = model.test_epoch_end(
        all_step_outputs=all_step_outputs,
        log_scores=False,
        verbose=True,
        save_csv_path=save_csv_path,
        keys_added_to_scores=getattr(args, 'keys_added_to_scores', []),
    )

    if args.save_detailed_scores_path:
        os.makedirs(os.path.dirname(args.save_detailed_scores_path), exist_ok=True)
        with open(args.save_detailed_scores_path, 'w') as wf:
            json.dump(detail_scores, wf)

    if only_return_pred_captions:
        return pred_captions

    if return_details:
        return scores, detail_scores
    
    return scores


def loop_n_frames(args, model, device):
    opt = model.get_opt()
    for i in range(1, opt['n_frames']+1):
        loader = get_loader({**opt, 'n_frames': i}, 'test', print_info=True if i == 1 else False, specific=args.specific, 
            not_shuffle=True, is_validation=True
        )
        scores = run_eval(args, model, loader, device)
        scores['n_frames'] = i
        scores['scope'] = opt['scope']
        scores['seed'] = opt['seed']
        save_dict_to_csv('./results_loop/', "n_frames.csv", scores)


def loop_category(args, model, device):
    opt = model.get_opt()
    assert opt['dataset'] == 'MSRVTT'

    for i in range(20):
        loader = get_loader(opt, 'test', print_info=True if i == 1 else False, specific=i, 
            not_shuffle=True, is_validation=True
        )
        scores = run_eval(args, model, loader, device)
        scores['category'] = i
        scores['scope'] = opt['scope']
        save_dict_to_csv('./results_loop/', "category.csv", scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='translate.py')
    parser.add_argument('-cp', '--checkpoint_paths', type=str, nargs='+', required=True)
    parser.add_argument('--base_data_path', type=str)

    cs = parser.add_argument_group(title='Common Settings')
    cs.add_argument('-gpus', '--gpus', type=int, default=1)
    cs.add_argument('-num_workers', '--num_workers', type=int, default=1)
    cs.add_argument('-fast', '--fast', default=False, action='store_true', 
        help='directly use Trainer.test()')
    cs.add_argument('-v', '--verbose', default=False, action='store_true',
        help='print some intermediate information (works when `fast` is False)')
    cs.add_argument('--save_csv', default=False, action='store_true',
        help='save result to csv file in model path (works when `fast` is False)')
    cs.add_argument('--csv_path', type=str)
    cs.add_argument('--csv_name', type=str, default='test_result.csv')

    ds = parser.add_argument_group(title='Dataloader Settings')
    ds.add_argument('-bsz', '--batch_size', type=int, default=128)
    ds.add_argument('-mode', '--mode', type=str, default='test',
        help='which set to run?', choices=['train', 'validate', 'test', 'all'])
    ds.add_argument('-specific', '--specific', default=-1, type=int, 
        help='run on the data of the specific category (only works in the MSR-VTT)')

    ar = parser.add_argument_group(title='Autoregressive Decoding Settings')
    ar.add_argument('-bs', '--beam_size', type=int, default=5, help='Beam size')
    ar.add_argument('-ba', '--beam_alpha', type=float)
    ar.add_argument('-topk', '--topk', type=int, default=1)

    na = parser.add_argument_group(title='Non-Autoregressive Decoding Settings')
    na.add_argument('-i', '--iterations', type=int, default=5)
    na.add_argument('-lbs', '--length_beam_size', type=int, default=6)
    na.add_argument('-q', '--q', type=int, default=1)
    na.add_argument('-qi', '--q_iterations', type=int, default=1)
    na.add_argument('-paradigm', '--paradigm', type=str, default='mp', choices=['mp', 'ef', 'l2r'])
    na.add_argument('-use_ct', '--use_ct', default=False, action='store_true')
    na.add_argument('-md', '--masking_decision', default=False, action='store_true')
    na.add_argument('-ncd', '--no_candidate_decision', default=False, action='store_true')
    na.add_argument('--algorithm_print_sent', default=False, action='store_true')
    na.add_argument('--teacher_path', type=str, default='')

    ts = parser.add_argument_group(title='Task Settings')
    ts.add_argument('-latency', '--latency', default=False, action='store_true', 
        help='batch_size will be set to 1 to compute the latency, which will be saved to latency.txt in the checkpoint folder')
    ts.add_argument('-sl', '--save_latency', action='store_true')
    
    parser.add_argument('-json_path', '--json_path', type=str, default='')
    parser.add_argument('-json_name', '--json_name', type=str, default='')
    parser.add_argument('-ns', '--no_score', default=False, action='store_true')
    parser.add_argument('-analyze', default=False, action='store_true')
    parser.add_argument('-collect_path', type=str, default='./collected_captions')
    parser.add_argument('-collect', default=False, action='store_true')
    parser.add_argument('-nobc', '--not_only_best_candidate', default=False, action='store_true')
            
    parser.add_argument('--attr', default=False, action='store_true')
    parser.add_argument('--probs_scaler', type=float, default=0.1)
    
    parser.add_argument('--with_backbones', type=str, nargs='+', default=[])
    parser.add_argument('--loop_n_frames', default=False, action='store_true')
    parser.add_argument('--loop_category', default=False, action='store_true')
    parser.add_argument('--calculate_mAP', default=False, action='store_true')
    parser.add_argument('--save_AP_path', type=str)
    parser.add_argument('--save_detailed_scores_path', type=str)

    parser.add_argument('--decoding_type', type=str)

    parser.add_argument('--wrapper', type=str, default='Model')

    parser.add_argument('--retrieval_topk', type=int)
    parser.add_argument('-rds', '--retrieval_datasets', type=str, nargs='+')
    parser.add_argument('--retrieval_db_ratio', type=float, default=100)

    parser.add_argument('--keys_added_to_scores', type=str, nargs='+', default=['seed'])
    args = parser.parse_args()

    # define the model
    model = load_model_from_arguments(args, ignore_empty_attributes=[
        'teacher_path', 'decoding_type', 'beam_alpha', 'retrieval_topk'])
    
    # get the device
    device = next(model.parameters()).device

    if args.loop_n_frames:
        loop_n_frames(args, model, device)
        sys.exit(0)
    elif args.loop_category:
        loop_category(args, model, device)
        sys.exit(0)
    
    if args.latency:
        args.batch_size = 1
    
    # this is used to check if using backbones to yield features has identical performance to the pre-extracted features
    image_preprocess_func = None
    if hasattr(model.captioner, 'backbone') and model.captioner.backbone is not None:
        image_preprocess_func = model.captioner.backbone.get_preprocess_func('i')
    
    # define the dataloader
    loader = get_loader(model.get_opt(), args.mode, print_info=True, specific=args.specific, 
        not_shuffle=True, batch_size=args.batch_size, is_validation=True, image_preprocess_func=image_preprocess_func, all_caps=getattr(args, 'all_caps', False)
    )

    if args.fast:
        # directly call pytorch-lightning's test loop
        # it will automatically decide where to run based on the gpu numbers (args.gpus)
        trainer = Trainer(logger=False, gpus=args.gpus)
        trainer.test(model, loader)
    else:
        # design the evaluation procedure on your own
        print(model)
        print(model.get_opt()['retrieval_topk'])
        print(f"Total Params: {sum(p.numel() for p in model.parameters())}")
        print(f"Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        run_eval(args, model, loader, device)

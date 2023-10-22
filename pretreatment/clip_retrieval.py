import sys
sys.path.append('..')
sys.path.append('.')
import torch
import clip
import os
import argparse
import pickle
import h5py
from config import Constants
from tqdm import tqdm
from dataloader import get_ids_set
from misc.utils import get_uniform_ids_from_k_snippets
import numpy as np
import time

def get_root(args):
    return os.path.join(Constants.base_data_path, args.dataset)

def get_feats_save_path(args, root=None):
    if root is None:
        root = get_root(args)
    
    args.arch = args.arch.replace('/', '-')
    
    fn = ['CLIP', args.arch, args.postfix]
    fn = '_'.join(fn) + '.hdf5'
    image_embs_path = os.path.join(root, 'feats', fn)
    text_embs_path = os.path.join(root, 'text_embs', fn)
    if args.arch_f:
        args.arch_f = args.arch_f.replace('/', '-')
        if not ('bert' in args.arch_f.lower() or 'glove' in args.arch_f.lower()):
            fn = 'CLIP_{}.hdf5'.format(args.arch_f)
        else:
            fn = '{}.hdf5'.format(args.arch_f)
        text_embs_path2 = os.path.join(root, 'text_embs', fn)
        assert os.path.exists(text_embs_path2), text_embs_path2
    else:
        text_embs_path2 = None
    
    assert os.path.exists(image_embs_path), image_embs_path
    assert os.path.exists(text_embs_path), text_embs_path

    return image_embs_path, text_embs_path, text_embs_path2


def run(image_features, text_features, topk, info=None, unique=False, refs=None, sampled_indices=None):
    assert image_features.shape[0] == 1
    if unique:
        assert refs is not None

    if sampled_indices is not None:
        logits_per_image = image_features @ text_features[sampled_indices].t()
        _, indices = logits_per_image.sort(-1, descending=True)
        indices = sampled_indices[indices.squeeze().cpu().numpy()]
    else:
        logits_per_image = image_features @ text_features.t()
        _, indices = logits_per_image.sort(-1, descending=True)
        indices = indices.squeeze().cpu().numpy()

    if info is not None:
        start, end = info
    success = 0
    relevant_ids = []
    unique_caps = set()

    for ind in indices:
        if info is not None and start <= ind < end:
            continue
        
        if unique:
            cap = refs[ind]
            if cap in unique_caps:
                continue
            unique_caps.add(cap)

        relevant_ids.append(ind)
        success += 1

        if success == topk:
            break
    
    return relevant_ids


def load(args, device, video_mode='all', text_mode='train'):
    def load_keys(dataset, mode='train'):
        info_corpus_path = os.path.join(Constants.base_data_path, dataset, 'info_corpus.pkl')
        split = pickle.load(open(info_corpus_path, 'rb'))['info']['split']
        return ['video%d'%_ for _ in get_ids_set(mode, split, is_vatex_activate=(dataset=='VATEX'))]

    args.arch = args.arch.replace('/', '-')
    # -------------- video ---------------
    vdb_path = os.path.join(
        Constants.base_data_path, 
        args.dataset, 
        'feats',
        '_'.join(['CLIP', args.arch] + ([args.postfix] if args.postfix else [])) + '.hdf5',
    )
    print('- Loading video features from', vdb_path)
    vdb = h5py.File(vdb_path, 'r')
    video_keys = load_keys(args.dataset, mode=video_mode)

    image_embs = []
    image_ids = get_uniform_ids_from_k_snippets(Constants.n_total_frames, args.n_frames)

    for key in tqdm(video_keys):
        i_embs = torch.from_numpy(np.asarray(vdb[key])) # (60, d)
        i_embs = i_embs[image_ids, :].mean(0) # (d, )
        image_embs.append(i_embs)
        
    # --------------  text  ---------------
    datasets = getattr(args, 'datasets', None)
    if datasets is None:
        datasets = [args.dataset]

    recorder, start_idx = [], 0
    text_embs, text_embs_e = [], []
    refs = []
    for i, dataset in enumerate(datasets):
        tdb_path = os.path.join(
            Constants.base_data_path, 
            dataset, 
            'text_embs',
            '_'.join(['CLIP', args.arch] + ([args.postfix] if args.postfix else [])) + '.hdf5',
        )
        print('- Loading text features from', tdb_path)
        tdb = h5py.File(tdb_path, 'r')
        text_keys = load_keys(dataset, mode=text_mode)

        if args.arch_f:
            args.arch_f = args.arch_f.replace('/', '-')
            if not ('bert' in args.arch_f.lower() or 'glove' in args.arch_f.lower()):
                fn = 'CLIP_{}.hdf5'.format(args.arch_f)
            else:
                fn = '{}.hdf5'.format(args.arch_f)

            tdb_e_path = os.path.join(
                Constants.base_data_path, 
                dataset, 
                'text_embs',
                fn,
            )
            print('- Loading text features (which truly embeds captions) from', tdb_e_path)
            tdb_e = h5py.File(tdb_e_path, 'r')
        else:
            tdb_e = None

        refs_data = pickle.load(open(os.path.join(Constants.base_data_path, dataset, 'refs.pkl'), 'rb'))
        for key in tqdm(text_keys):
            t_embs = torch.from_numpy(np.asarray(tdb[key])) # (n_captions, d)
            text_embs.append(t_embs)

            if tdb_e is not None:
                text_embs_e.append(torch.from_numpy(np.asarray(tdb_e[key])))

            if dataset == args.dataset:
                n_captions = t_embs.shape[0]
                recorder.append((start_idx, start_idx + n_captions))
                start_idx += n_captions
            
            for item in refs_data[key]:
                refs.append(item['caption'])

    image_embs = torch.stack(image_embs, 0).to(device) # (n_total_videos, d)
    text_embs = torch.cat(text_embs, dim=0).to(device) # (n_total_captions, d)

    if len(text_embs_e):
        text_embs_e = torch.cat(text_embs_e, dim=0)
    else:
        text_embs_e = None

    print('- Shpae of image embs:', image_embs.shape)
    print('- Shpae of text embs:', text_embs.shape)
    image_features = image_embs / image_embs.norm(dim=-1, keepdim=True)
    text_features = text_embs / text_embs.norm(dim=-1, keepdim=True)

    all_indices = [_ for _ in range(text_embs.size(0))]
    if args.ratio < 100:
        n_sampled_capions = int(text_embs.size(0) * args.ratio / 100)
        print('- Only keep {} ({:.1f}%) {} captions'.format(n_sampled_capions, args.ratio, text_mode))
        import random
        random.seed(0)
        sampled_indices = np.array(sorted(random.sample(all_indices, n_sampled_capions)))
    else:
        print('- Keep all {} captions for retrieval'.format(text_mode))
        sampled_indices = None

    return video_keys, image_features, text_features, text_embs, text_embs_e, recorder, refs, sampled_indices


def run_retrieval(args, device):
    assert args.dataset == 'MSRVTT'

    video_keys, image_features, text_features, text_embs, text_embs_e, recorder, refs, sampled_indices = load(args, device, args.eval_mode, args.eval_mode)    

    all_rank = []
    all_precision = []
    
    K_list = [1, 5, 10]
    all_recall_K = [[] for _ in K_list]

    for i, vid in tqdm(enumerate(video_keys)):
        logits_per_image = image_features[i, :].unsqueeze(0) @ text_features.t()
        _, indices = logits_per_image.sort(-1, descending=True)
        _, rank = indices.sort(dim=1)

        start, end = recorder[i]
        gt_captions_rank = rank.float().view(-1)[start:end]
        first_relevant_rank = gt_captions_rank.min().item() + 1
        all_rank.append(first_relevant_rank)

        for j, K in enumerate(K_list):
            has_at_least_one_relevant = gt_captions_rank.lt(K).sum().item() > 0
            if has_at_least_one_relevant:
                all_recall_K[j].append(1.0)
            else:
                all_recall_K[j].append(0.0)

        positive_label = torch.arange(start, end).to(device) # [n_positive_labels]
        hit_rank = rank.view(-1)[positive_label]
        sorted_hit_rank, _ = hit_rank.sort()
        ids = torch.arange(len(positive_label)).to(device)

        precision = (ids + 1) / (sorted_hit_rank + 1)
        average_precision = precision.mean().cpu().item()
        all_precision.append(average_precision)

    for j, K in enumerate(K_list):
        print('Recall@%d:'%K, np.mean(all_recall_K[j]) * 100)

    print('Mean Average Precision:', np.mean(all_precision) * 100)
    print('Median Rank:', np.median(all_rank))
    print('Mean Rank:', np.mean(all_rank))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='MSRVTT', choices=['MSRVTT', 'MSVD', 'VATEX'])
    parser.add_argument('-ds', '--datasets', type=str, nargs='+')
    parser.add_argument('-arch', '--arch', type=str, default='ViT-B/32', choices=clip.available_models())
    parser.add_argument('-arch_f', '--arch_f', type=str)
    parser.add_argument('-no_cuda', '--no_cuda', default=False, action='store_true')
    parser.add_argument('-nf', '--n_frames', type=int, default=28)
    parser.add_argument('-topk', '--topk', type=int, default=100)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval_mode', type=str, default='test')
    parser.add_argument('--unique', action='store_true')
    parser.add_argument('--postfix', type=str, default='')
    parser.add_argument('--ratio', type=float, default=100, 
                        help='the ratio of training captions to constitute the retrieval database, default to 100')

    parser.add_argument('--retrieval_database', type=str, nargs='+', default=[], 
        help='which datasets\' corpora is treated as the retrieval database? '
        'E.g., [] (detemined by `--dataset`), [MSRVTT], [VATEX], [MSRVTT, VATEX]')

    parser.add_argument('--latency', action='store_true')
    parser.add_argument('--n_latency_samples', type=int, default=1000)
    parser.add_argument('--n_latency_captions', type=int)
    args = parser.parse_args()

    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    if args.eval:
        run_retrieval(args, device)
    else:
        video_keys, image_features, text_features, text_embs, text_embs_e, recorder, refs, sampled_indices = load(args, device)    

        print('- Start retrieval')

        if args.latency:
            total_time = 0
            video_keys = video_keys[:args.n_latency_samples]
            if args.n_latency_captions:
                D = text_features.size(1)
                text_embs_e = None
                text_embs = torch.randn(args.n_latency_captions, D).to(device)
                text_features = text_embs / text_embs.norm(dim=-1, keepdim=True)
                print(f'- Calcualting latency of retrieving {args.n_latency_captions} captions')
        else:
            save_path = os.path.join(Constants.base_data_path, args.dataset, 'retrieval')
            os.makedirs(save_path, exist_ok=True)

            fn = ['CLIP', args.arch]
            if args.postfix:
                fn.append(args.postfix)
            if args.arch_f:
                fn.append(args.arch_f)
            if args.datasets is not None:
                fn.append('-'.join(args.datasets))
            if args.unique:
                fn.append('unique')
            if args.ratio < 100:
                fn.append('ratio%.1f' % args.ratio)

            fn = '_'.join(fn) + '.hdf5'

            save_path = os.path.join(save_path, fn)
            print('- Saving relevant info to', save_path)
            db = h5py.File(save_path, 'a')

        for i, key in tqdm(enumerate(video_keys)):
            if args.latency:
                start_time = time.time()

            relevant_ids = run(
                image_features[i, :].unsqueeze(0), 
                text_features,
                info=recorder[i] if i < len(recorder) else (1e9, 1e9),
                topk=args.topk,
                unique=args.unique,
                refs=refs,
                sampled_indices=sampled_indices,
            )
            if text_embs_e is not None:
                relevant_embs = text_embs_e[relevant_ids, :].cpu().numpy()
            else:
                relevant_embs = text_embs[relevant_ids, :].cpu().numpy()
            
            if args.latency:
                total_time += (time.time() - start_time)
            else:
                db[key] = relevant_embs
                db[key+'_i'] = np.array(relevant_ids)

        if args.latency:
            print(f'- # samples: {len(video_keys)}')
            print(f'- Total inference time: {total_time}')
            print(f'- Average latency: {total_time / len(video_keys)}')
            with open('latency.txt', 'a') as f:
                f.write(f'retrieval-{args.arch}\t{len(video_keys)}\t{total_time / len(video_keys)}\t{text_embs.shape}\n')
        else:
            if sampled_indices is not None:
                db['sampled_indices'] = sampled_indices
            db.close()

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


def get_root(args):
    return os.path.join(Constants.base_data_path, args.dataset)

def get_feats_save_path(args, root=None):
    if root is None:
        root = get_root(args)
    
    args.arch = args.arch.replace('/', '-')
    feats_save_path = os.path.join(root, 'text_embs')
    os.makedirs(feats_save_path, exist_ok=True)

    fn = ['CLIP', args.arch, args.postfix]
    fn = '_'.join(fn) + '.hdf5'
    feats_save_path = os.path.join(feats_save_path, fn)

    return feats_save_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='MSRVTT', choices=['MSRVTT', 'MSVD', 'VATEX'])
    parser.add_argument('-arch', '--arch', type=str, default='ViT-B/32', choices=clip.available_models())
    parser.add_argument('-no_cuda', '--no_cuda', default=False, action='store_true')
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--postfix', type=str, default='')
    args = parser.parse_args()

    if args.no_cuda or not torch.cuda.is_available():
        device = 'cpu'
    else:
        device = 'cuda'

    model, preprocess = clip.load(args.arch, device=device, jit=False)

    if args.checkpoint_path:
        print(f'- Loading checkpoint from {args.checkpoint_path}')
        state_dict = torch.load(args.checkpoint_path, 'cpu')['state_dict']
        if 'logit_scale' not in state_dict:
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print(model.logit_scale)

    model.float()
    model.eval()
    model.to(device)
    
    root = os.path.join(Constants.base_data_path, args.dataset)

    info_corpus_path = os.path.join(root, 'info_corpus.pkl')
    split = pickle.load(open(info_corpus_path, 'rb'))['info']['split']
    all_video_ids = get_ids_set('all', split, is_vatex_activate=(args.dataset=='VATEX'))

    refs_path = os.path.join(root, 'refs.pkl')
    data = pickle.load(open(refs_path, 'rb'))

    save_path = get_feats_save_path(args, root)
    print('- Save all feats to {}'.format(save_path))
    db = h5py.File(save_path, 'a')

    for id in tqdm(all_video_ids):
        vid = 'video%d'%id
        if vid in db.keys():
            continue

        this_data = data[vid]
        captions = [item['caption'] for item in this_data]
        captions = clip.tokenize(captions, truncate=True).to(device)

        with torch.no_grad():
            text_embs = model.encode_text(captions)
        
        print(vid, text_embs.shape, text_embs.min(), text_embs.max(), text_embs.mean(), text_embs.std())
        db[vid] = text_embs.cpu().numpy()

    db.close()

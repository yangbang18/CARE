import sys
sys.path.append('..')
sys.path.append('.')
import torch
import clip
import os
import argparse
import pickle
import h5py
import glob
import time
from config import Constants
from PIL import Image
from tqdm import tqdm
from misc.utils import get_uniform_items_from_k_snippets
from dataloader import get_ids_set
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class FrameDataset(Dataset):
    def __init__(self, video_ids, all_frames_path, frames_suffix, preprocess, n_frames=None) -> None:
        super().__init__()
        self.video_ids = video_ids
        self.all_frames_path = all_frames_path
        self.frames_suffix = frames_suffix
        self.preprocess = preprocess
        self.n_frames = n_frames if n_frames is not None else Constants.n_total_frames
    
    def __len__(self):
        return len(self.video_ids)
    
    def __getitem__(self, index):
        vid = 'video%d'%self.video_ids[index]

        frames = sorted(glob.glob(os.path.join(self.all_frames_path, vid, '*.{}'.format(self.frames_suffix))))
        frames = get_uniform_items_from_k_snippets(frames, self.n_frames) # uniformly sampling e.g. 60 frames
        images_of_this_vid = [self.preprocess(Image.open(f)) for f in frames] # preprocess and transform these sampled frames
        images_of_this_vid = torch.stack(images_of_this_vid, dim=0)

        return vid, images_of_this_vid


def get_loader(num_workers=8, **kwargs):
    return DataLoader(
        FrameDataset(**kwargs), 
        batch_size=1, 
        shuffle=False,
        num_workers=num_workers,
    )


@torch.no_grad()
def latency(model, preprocess, device, all_frames_path, frames_suffix, video_ids, n_frames, num_workers):
    model.float()
    model.eval()
    model.to(device)

    loader = get_loader(
        num_workers=num_workers,
        video_ids=video_ids,
        all_frames_path=all_frames_path,
        frames_suffix=frames_suffix,
        preprocess=preprocess,
        n_frames=n_frames,
    )
    
    total_time = 0
    
    for vid, images in tqdm(loader):
        assert len(vid) == 1
        assert len(images) == 1
        
        vid = vid[0]
        images = images.squeeze(0).to(device)
        assert images.size(0) == n_frames

        start_time = time.time()
        _ = model.encode_image(images)
        total_time += (time.time() - start_time)
    
    print(f'- # samples: {len(loader)}')
    print(f'- Total inference time: {total_time}')
    print(f'- Average latency: {total_time / len(loader)}')
    return total_time / len(loader)


@torch.no_grad()
def prepare_encoded_image_feats(model, preprocess, device, all_frames_path, frames_suffix, video_ids, db, num_workers):
    # the original weights of CLIP have the type of torch.float16 (model.half())
    # changing it to torch.float32 (model.float()) is important to get the identical captioning performance
    model.float()
    model.eval()
    model.to(device)

    loader = get_loader(
        num_workers=num_workers,
        video_ids=[id for id in video_ids if 'video%d'%id not in db],
        all_frames_path=all_frames_path,
        frames_suffix=frames_suffix,
        preprocess=preprocess
    )
    
    for vid, images in tqdm(loader):
        assert len(vid) == 1
        assert len(images) == 1

        vid = vid[0]
        images = images.squeeze(0).to(device)
        image_feats_of_this_vid = model.encode_image(images)
        if image_feats_of_this_vid.dim() > 2:
            image_feats_of_this_vid = image_feats_of_this_vid.squeeze()

        print(vid, image_feats_of_this_vid.shape)
        db[vid] = image_feats_of_this_vid.cpu().numpy()


def get_root(args):
    return os.path.join(Constants.base_data_path, args.dataset)

def get_feats_save_path(args, root=None):
    if root is None:
        root = get_root(args)
    
    args.arch = args.arch.replace('/', '-')
    feats_save_path = os.path.join(root, 'feats')
    os.makedirs(feats_save_path, exist_ok=True)

    fn = ['CLIP', args.arch] + ([args.postfix] if args.postfix else [])
    if args.replace:
        fn.append('rep')
    fn = '_'.join(fn) + '.hdf5'
    feats_save_path = os.path.join(feats_save_path, fn)

    return feats_save_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='MSRVTT', choices=['MSRVTT', 'MSVD', 'VATEX'])
    parser.add_argument('-fp', '--all_frames_path', type=str, default='')
    parser.add_argument('-fs', '--frames_suffix', type=str, default='jpg')
    parser.add_argument('-arch', '--arch', type=str, default='ViT-B/32', choices=clip.available_models())
    parser.add_argument('-no_cuda', '--no_cuda', default=False, action='store_true')
    parser.add_argument('-replace', '--replace', default=False, action='store_true')
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--postfix', type=str, default='')
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--latency', action='store_true')
    parser.add_argument('--n_frames', type=int, default=28)
    parser.add_argument('--n_latency_samples', type=int, default=1000)
    args = parser.parse_args()

    if args.no_cuda or not torch.cuda.is_available():
        device = 'cpu'
    else:
        device = 'cuda'

    model, preprocess = clip.load(args.arch, device=device, jit=False)
    if args.replace:
        model.visual.attnpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))

    if args.checkpoint_path:
        print(f'- Loading checkpoint from {args.checkpoint_path}')
        state_dict = torch.load(args.checkpoint_path, 'cpu')['state_dict']
        if 'logit_scale' not in state_dict:
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print(model.logit_scale)
    
    if not args.all_frames_path:
        args.all_frames_path = os.path.join(Constants.base_data_path, args.dataset, 'all_frames')
    assert os.path.exists(args.all_frames_path)
    
    root = os.path.join(Constants.base_data_path, args.dataset)
    
    info_corpus_path = os.path.join(root, 'info_corpus.pkl')
    split = pickle.load(open(info_corpus_path, 'rb'))['info']['split']
    all_video_ids = get_ids_set('all', split, is_vatex_activate=(args.dataset=='VATEX'))

    if args.latency:
        all_video_ids = all_video_ids[:args.n_latency_samples]
        t = latency(
            model, 
            preprocess, 
            device, 
            args.all_frames_path, 
            args.frames_suffix, 
            all_video_ids, 
            args.n_frames,
            args.num_workers,
        )
        with open('latency.txt', 'a') as f:
            f.write(f'visual-{args.arch}\t{len(all_video_ids)}\t{t}\n')
    else:
        feats_save_path = get_feats_save_path(args, root)
        print('- Save all feats to {}'.format(feats_save_path))
        
        db = h5py.File(feats_save_path, 'a')
        prepare_encoded_image_feats(
            model, 
            preprocess, 
            device, 
            args.all_frames_path, 
            args.frames_suffix, 
            all_video_ids, 
            db,
            args.num_workers
        )
        db.close()

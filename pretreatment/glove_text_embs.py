import sys
sys.path.append('..')
sys.path.append('.')
from config import Constants
from dataloader import get_ids_set

import argparse
import os
import torch
import pandas as pd
import pickle
from tqdm import tqdm
import h5py
from collections import Counter
import numpy as np

def prepare_pretrained_word_embeddings(words, dim=300, pretrained_path='/home/yangbang/new_VC_data/glove.840B.300d.txt'):
    itow = {i: w for i, w in enumerate(words.keys())}
    wtoi = {v: k for k, v in itow.items()}

    embs = np.zeros((len(words), dim))
    print('- Loading pretrained word embeddigns from {}'.format(pretrained_path))
    num_existed = 0
    num_lines_have_read = 0
    visit = np.zeros(len(words))
    with open(pretrained_path, 'r') as f:
        while True:
            line = f.readline().strip()
            num_lines_have_read += 1

            if not line:
                break

            content = line.split()
            num = len(content) - dim
            w = '-'.join(content[:num])
            if w in words:
                assert not visit[wtoi[w]], f"{content}, {len(content)}, {w}, {wtoi[w]}, {itow[wtoi[w]]}"
                num_existed += 1
                embs[wtoi[w]] = np.array([float(i) for i in content[num:]])
                print('- Have read {} lines, {}/{} words exist, {}'.format(
                    num_lines_have_read, num_existed, len(words), w))
                visit[wtoi[w]] = 1

    print('- The number of total lines: {}, {}/{} words exist'.format(
        num_lines_have_read, num_existed, len(words)))

    print('- Words below can not be found in pretrained word embeddings (all initiliazed to zero vectors):')
    print([itow[i] for i, flag in enumerate(visit) if not flag])

    return embs, visit, itow, wtoi


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default='MSRVTT')
parser.add_argument('--mode', type=str, default='mean')
args = parser.parse_args()

root = os.path.join(Constants.base_data_path, args.dataset)

info_corpus_path = os.path.join(root, 'info_corpus.pkl')
split = pickle.load(open(info_corpus_path, 'rb'))['info']['split']
train_video_ids = get_ids_set('train', split, is_vatex_activate=(args.dataset=='VATEX'))

refs_path = os.path.join(root, 'refs.pkl')
data = pickle.load(open(refs_path, 'rb'))

print('- Preparing glove embs')
words = []
for id in train_video_ids:
    vid = 'video%d'%id
    for item in data[vid]:
        words.extend(item['caption'].split(' '))
words = Counter(words)
embs, visit, itow, wtoi = prepare_pretrained_word_embeddings(words)

all_video_ids = get_ids_set('all', split, is_vatex_activate=(args.dataset=='VATEX'))
feats_save_path = os.path.join(root, 'text_embs')
os.makedirs(feats_save_path, exist_ok=True)
feats_save_path = os.path.join(feats_save_path, 'glove.hdf5' if args.mode == 'mean' else 'glove_max.hdf5')

print('- Save all feats to {}'.format(feats_save_path))
db = h5py.File(feats_save_path, 'a')

for id in tqdm(all_video_ids):
    vid = 'video%d'%id
    if vid in db.keys():
        continue

    feats = []
    for item in data[vid]:
        this_feat = []
        for w in item['caption'].split(' '):
            if w not in words or not visit[wtoi[w]]:
                continue
            this_feat.append(embs[wtoi[w]])
        
        if len(this_feat):
            this_feat = torch.from_numpy(np.array(this_feat))
            if args.mode == 'mean':
                this_feat = this_feat.mean(0).numpy()
            else:
                this_feat = this_feat.max(0)[0].numpy()
        else:
            this_feat = np.zeros(300)

        feats.append(this_feat)

    feats = np.array(feats)
    print(feats.shape)
    db[vid] = feats

db.close()

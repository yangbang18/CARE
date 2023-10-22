import sys
sys.path.append('..')
sys.path.append('.')
from config import Constants
from dataloader import get_ids_set
from tqdm import tqdm
from transformers import BertTokenizer, AutoModel

import argparse
import os
import torch
import pickle
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default='MSRVTT')
parser.add_argument('--mode', type=str, default='mean')
args = parser.parse_args()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

model.eval()
model.to(device)

root = os.path.join(Constants.base_data_path, args.dataset)

info_corpus_path = os.path.join(root, 'info_corpus.pkl')
split = pickle.load(open(info_corpus_path, 'rb'))['info']['split']
all_video_ids = get_ids_set('all', split, is_vatex_activate=(args.dataset=='VATEX'))

refs_path = os.path.join(root, 'refs.pkl')
data = pickle.load(open(refs_path, 'rb'))

feats_save_path = os.path.join(root, 'text_embs')
os.makedirs(feats_save_path, exist_ok=True)
feats_save_path = os.path.join(feats_save_path, 'BERT.hdf5' if args.mode == 'mean' else 'BERT_max.hdf5')

print('- Save all feats to {}'.format(feats_save_path))
db = h5py.File(feats_save_path, 'a')


for id in tqdm(all_video_ids):
    vid = 'video%d'%id
    if vid in db.keys():
        continue

    this_data = data[vid]
    captions = [item['caption'] for item in this_data]

    lens = []
    for cap in captions:
        lens.append(len(tokenizer(cap)['input_ids']) - 2)
    print(lens)

    inputs = tokenizer(captions, return_tensors="pt", padding=True, truncation=True)
    for k in inputs.keys():
        inputs[k] = inputs[k].to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs['hidden_states'][-1]

    feats = []
    for h, length in zip(hidden_states, lens):
        if args.mode == 'mean':
            feats.append(h[1:1+length, :].mean(dim=0))
        else:
            feats.append(h[1:1+length, :].max(dim=0)[0])
    
    feats = torch.stack(feats, dim=0)
    print(vid, feats.shape)
    db[vid] = feats.cpu().numpy()

db.close()

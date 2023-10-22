import wget
import pickle
import json
import string
import nltk
import os
import numpy as np
from tqdm import tqdm
from config import Constants
from collections import defaultdict


def preprocess_MSVD(base_path):
    os.makedirs(base_path, exist_ok=True)

    # the official url is https://www.cs.utexas.edu/users/ml/clamp/videoDescription
    # but we found that the download link no longer available
    url = "https://github.com/ybCliff/VideoCaptioning/releases/download/1.0/msvd_refs.pkl"
    refs_pickle = os.path.join(base_path, 'refs.pkl')
    if not os.path.exists(refs_pickle):
        wget.download(url, out=refs_pickle)
    
    mapping_info = open('data/msvd_mapping.txt', 'r').read().strip().split('\n')
    
    vid2id = {}
    for line in mapping_info:
        _id, vid = line.split()
        vid2id[vid] = _id

    split = {
        'train': [i for i in range(1200)],
        'validate': [i for i in range(1200, 1300)],
        'test': [i for i in range(1300, 1970)]
    }

    raw_caps_all = defaultdict(list)
    raw_caps_train = {}
    
    refs = pickle.load(open(refs_pickle, 'rb'))
    for vid in tqdm(refs.keys()):
        num = int(vid[5:]) # e.g. 'video999', num = 999
        for item in refs[vid]:
            tokens = item['caption'].lower().split()
            raw_caps_all[vid].append(tokens)

        if num in split['train']:
            raw_caps_train[vid] = raw_caps_all[vid]

    return {
        'split': split, 
        'raw_caps_train': raw_caps_train, 
        'raw_caps_all': raw_caps_all, 
        'vid2id': vid2id,
    }


def preprocess_MSRVTT(base_path):
    os.makedirs(base_path, exist_ok=True)
    # the official url is http://ms-multimedia-challenge.com/2016
    url = "https://github.com/ybCliff/VideoCaptioning/releases/download/v1.0/videodatainfo_2016.json"
    input_json = os.path.join(base_path, 'videodatainfo.json')
    if not os.path.exists(input_json):
        wget.download(url, out=input_json)

    json_data = json.load(open(input_json, 'r'))
    sentences = json_data['sentences']
    videos = json_data['videos']

    split = {'train': [], 'validate': [], 'test': []}
    for v in videos:
        split[v['split']].append(int(v['id']))

    raw_caps_all = defaultdict(list)
    raw_caps_train = defaultdict(list)
    references = defaultdict(list)

    for item in tqdm(sentences):
        vid = item['video_id']
        tokens = [
            token.lower() \
            for token in item['caption'].split() \
            if token not in string.punctuation
        ]

        raw_caps_all[vid].append(tokens)

        if int(vid[5:]) in split['train']:
            raw_caps_train[vid].append(tokens)

        references[vid].append({
            'image_id': vid, 
            'cap_id': len(references[vid]), 
            'caption': ' '.join(tokens)
        })

    itoc = {}
    split_category = {'train': defaultdict(list), 'validate': defaultdict(list), 'test': defaultdict(list)}
    for item in videos:
        itoc[item["id"]] = item["category"]
        split_category[item['split']][int(item["category"])].append(int(item['id']))

    return {
        'split': split, 
        'raw_caps_train': raw_caps_train, 
        'raw_caps_all': raw_caps_all, 
        'references': references,
        'itoc': itoc,
        'split_category': split_category
    }


def preprocess_VATEX(base_path):
    os.makedirs(base_path, exist_ok=True)
    base_url = "https://eric-xw.github.io/vatex-website/data/"

    references = defaultdict(list)
    raw_caps_train = defaultdict(list)
    raw_caps_valid = defaultdict(list)
    raw_caps_test = defaultdict(list)
    split = {'train': [], 'validate': [], 'test': []}

    vid2id = get_vid2id_mappings('./data/vatex_mapping.txt')
    id2vid = {v: k for k, v in vid2id.items()}

    for mode, filename, raw_captions in zip(
            ['train', 'validate', 'test'], 
            ['vatex_training_v1.0.json', 'vatex_validation_v1.0.json', 'vatex_public_test_english_v1.1.json'],
            [raw_caps_train, raw_caps_valid, raw_caps_test]
        ):
        now_path = os.path.join(base_path, filename)
        now_url = base_url + filename
        print(now_url, now_path)
        if not os.path.exists(now_path):
            import wget
            wget.download(now_url, out=now_path)
        
        print('===> Loading `{}` data'.format(mode))
        data = json.load(open(now_path))
        
        for item in tqdm(data):
            _id = item['videoID'] # youtubeid_start_end
            vid = id2vid[_id] # e.g., video500
            split[mode].append(int(vid[5:])) # e.g., 'train' = [..., 500]
            
            for sent in item['enCap']:
                tokens = [
                    token.lower().replace('\'s', 's') \
                    for token in nltk.word_tokenize(sent) \
                    if (token not in string.punctuation and token)
                ] # change to lowercase letters; replace 's with s; remove commas and periods
                raw_captions[vid].append(tokens)
                references[vid].append({
                    'image_id': vid, 
                    'cap_id': len(references[vid]), 
                    'caption': ' '.join(tokens)
                })

    def get_vatex_activate_split(split, all_videos_path=None):
        if all_videos_path is None:
            all_videos_path = os.path.join(Constants.base_data_path, 'VATEX', 'all_videos')

        print('=' * 100)
        print('- check how many videos are available after downloading')
        print('- path of videos: {}'.format(all_videos_path))
        assert os.path.exists(all_videos_path)
        
        activate_vids = set()
        for video_file in os.listdir(all_videos_path):
            vid = video_file.split('.')[0]
            activate_vids.add(vid)
        print('- {} videos are avaliable!'.format(len(activate_vids)))
        
        for mode in ['train', 'validate', 'test']:
            activate_mode = 'activate_%s' % mode
            split[activate_mode] = []
            for idx in split[mode]:
                vid = 'video%d' % idx
                if vid in activate_vids:
                    split[activate_mode].append(idx)

        for mode in ['train', 'validate', 'test']:
            print('- the expected (exact) number of videos in `{}` set is {} ({})'.format(
                mode, len(split[mode]), len(split['activate_%s'%mode])
            ))
        print('=' * 100)

    get_vatex_activate_split(split)

    return {
        'split': split, 
        'raw_caps_train': raw_caps_train, 
        'raw_caps_all': {**raw_caps_train, **raw_caps_valid, **raw_caps_test}, 
        'references': references,
        'vid2id': vid2id, 
    }


def get_vid2id_mappings(path):
    info = open(path).read().strip().split('\n')
    vid2id = {}
    for line in info:
        _id, vid = line.split()
        vid2id[vid] = _id
    return vid2id


def build_vocab(train_vid2caps, count_thr, sort_vocab=False, attribute_first=False):
    '''
        args:
            - train_vid2caps (dict): vid-captions pairs from the training set
            - count_thr (int): words that appear <= count_thr will be filtered
        return:
            - vocab (list): vocabulary
    '''

    # count up the number of words
    counts = {}
    for vid, caps in train_vid2caps.items():
        for cap in caps:
            for w in cap:
                counts[w] = counts.get(w, 0) + 1

    bad_words = [w for w, n in counts.items() if n <= count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    total_words = sum(counts.values())

    print('- The number of bad words: %d/%d = %.2f%%' %
          (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts)))
    print('- The number of the vocabulary: %d' % (len(counts) - len(bad_words)))
    print('- The number of UNKs: %d/%d = %.2f%%' %
          (bad_count, total_words, bad_count * 100.0 / total_words))

    candidate_vocab = [(w, n) for w, n in counts.items() if n > count_thr]
    if sort_vocab:
        print('- Sort the vocabulary by the frequency of words, larger the first')
        candidate_vocab = sorted(candidate_vocab, key=lambda x: -x[1])

    if sort_vocab and attribute_first:
        num_attributes = Constants.attribute_end - Constants.attribute_start
        
        print('- %d attributes will be placed at the top of the vocab' % num_attributes)

        vocab = []
        stop_words_list = get_stop_words_list()
        is_stop_words = []
        index = 0
        for i, (w, n) in enumerate(candidate_vocab):
            if w in stop_words_list:
                is_stop_words.append(w)
            else:
                vocab.append(w)
                index += 1
                if index == num_attributes:
                    break
        
        num_attributes = min(len(vocab), num_attributes)
        vocab += is_stop_words
        vocab += [w for w, _ in candidate_vocab[i+1:]]
        
        non_stop_word_count = sum(counts[w] for w in vocab if w not in stop_words_list)
        for k in range(100, num_attributes, 100):
            attribute_count = sum(counts[w] for w in vocab[:k])
            print('- The ratio of top %d attributes over all attributes: %d/%d = %.2f%%' %
                (k, attribute_count, non_stop_word_count, attribute_count * 100.0 / non_stop_word_count))

        attribute_count = sum(counts[w] for w in vocab[:num_attributes])
        print('- The ratio of top %d attributes over all attributes: %d/%d = %.2f%%' %
                (num_attributes, attribute_count, non_stop_word_count, attribute_count * 100.0 / non_stop_word_count))
    else:
        vocab = [w for w, _ in candidate_vocab]

    assert len(vocab) == len(counts) - len(bad_words)

    print('- Top 100 words:')
    print(vocab[:100])
    return vocab


def get_length_info(captions):
    length_info = {}
    max_length = 50

    for vid, caps in captions.items():
        length_info[vid] = [0] * max_length
        for cap in caps:
            length = len(cap) - 2 # exclude <bos>, <eos>
            if length >= max_length:
                continue
            length_info[vid][length] += 1

    return length_info


def get_captions_and_pos_tags(raw_caps_all, vocab):
    itow = {i + 6: w for i, w in enumerate(vocab)}
    itow[Constants.PAD] = Constants.PAD_WORD
    itow[Constants.UNK] = Constants.UNK_WORD
    itow[Constants.BOS] = Constants.BOS_WORD
    itow[Constants.EOS] = Constants.EOS_WORD
    itow[Constants.MASK] = Constants.MASK_WORD
    itow[Constants.VIS] = Constants.VIS_WORD

    wtoi = {w: i for i, w in itow.items()}  # inverse table

    ptoi = {}
    ptoi[Constants.PAD_WORD] = Constants.PAD
    ptoi[Constants.UNK_WORD] = Constants.UNK
    ptoi[Constants.BOS_WORD] = Constants.BOS
    ptoi[Constants.EOS_WORD] = Constants.EOS
    ptoi[Constants.MASK_WORD] = Constants.MASK
    ptoi[Constants.VIS_WORD] = Constants.VIS
    tag_start_i = 6

    captions = defaultdict(list)
    pos_tags = defaultdict(list)
    for vid, caps in tqdm(raw_caps_all.items()):
        for cap in caps:
            tag_res = nltk.pos_tag(cap)

            caption_id = [Constants.BOS]
            tagging_id = [Constants.BOS]

            for w, t in zip(cap, tag_res):
                assert t[0] == w
                tag = Constants.pos_tag_mapping[t[1]]

                if w in wtoi.keys():
                    caption_id += [wtoi[w]]
                    if tag not in ptoi.keys():
                        ptoi[tag] = tag_start_i
                        tag_start_i += 1
                    tagging_id += [ptoi[tag]]
                else:
                    caption_id += [Constants.UNK]
                    tagging_id += [Constants.UNK]

            caption_id += [Constants.EOS]
            tagging_id += [Constants.EOS]

            captions[vid].append(caption_id)
            pos_tags[vid].append(tagging_id)

    itop = {i: t for t, i in ptoi.items()}
    return itow, captions, itop, pos_tags


def prepare_pretrained_word_embeddings(args, itow):
    wtoi = {v:k for k, v in itow.items()}
    num_total_words = len(itow)
    embs = np.zeros((num_total_words, args.pretrained_dim))

    print('- Loading pretrained word embeddigns from {}'.format(args.pretrained_path))
    num_existed = 0
    num_lines_have_read = 0
    visit = np.zeros(num_total_words)
    with open(args.pretrained_path, 'r') as f:
        while True:
            line = f.readline().strip()
            num_lines_have_read += 1

            if not line:
                break

            content = line.split()
            num = len(content) - args.pretrained_dim # some lines have more than one word
            w = '-'.join(content[:num])
            if w in wtoi:
                assert not visit[wtoi[w]], f"{content}, {len(content)}, {w}, {wtoi[w]}, {itow[wtoi[w]]}"
                num_existed += 1
                embs[wtoi[w]] = np.array([float(i) for i in content[num:]])
                print('- Have read {} lines, {}/{} words exist, {}'.format(
                    num_lines_have_read, num_existed, num_total_words, w))
                visit[wtoi[w]] = 1
    
    print('- The number of total lines: {}, {}/{} words exist'.format(
        num_lines_have_read, num_existed, num_total_words))
    
    print('- Words below can not be found in pretrained word embeddings (all initiliazed to zero vectors):')
    print([itow[i] for i, flag in enumerate(visit) if not flag])
    
    save_path = os.path.join(args.base_pth, args.save_name)
    print('- Word embeddings are saved to {}'.format(save_path))
    np.save(save_path, embs)


def prepare_category_embeddings(args):
    assert args.pretrained_path, \
        'path to the file that contains pretrained word embs should be specified'

    category2index = {}
    index2num = {}
    for index, category in Constants.index2category.items():
        all_category = category.split('/')
        for c in all_category:
            category2index[c] = index
        index2num[index] = len(all_category)

    embs = np.zeros((len(Constants.index2category), args.pretrained_dim))
    num_exists = 0

    print('- Loading pretrained word embeddigns from {}'.format(args.pretrained_path))
    with open(args.pretrained_path, 'r') as f:
        while True:
            line = f.readline().strip()
            if not line: 
                break

            content = line.split()
            num = len(content) - args.pretrained_dim # some lines have more than one word
            w = '-'.join(content[:num])
            if w in category2index:
                num_exists += 1
                print(w)
                embs[category2index[w]] += np.array([float(i) for i in content[num:]])
    
    assert num_exists == len(category2index)

    for i, num in index2num.items():
        embs[i] = embs[i] / num
    
    return embs


def get_vid2attribute_mappings(ids_set, captions):
    num_attributes = Constants.attribute_end - Constants.attribute_start
    vid2attr = {}
    for _id in ids_set:
        vid = 'video%d' % _id
        caps = captions[vid]
        attr_label = np.zeros(num_attributes)
        for cap in caps:
            assert cap[0] == Constants.BOS
            assert cap[-1] == Constants.EOS
            
            for word_id in cap[1:-1]:
                if Constants.attribute_start <= word_id < Constants.attribute_end:
                    attribute_id = word_id - Constants.attribute_start
                    attr_label[attribute_id] = 1
        
        vid2attr[vid] = attr_label
    return vid2attr


def get_attribute_embeddings(embeddings, topk):
    start_ids = Constants.attribute_start
    end_ids = start_ids + topk
    assert end_ids <= Constants.attribute_end, \
        "topk should be in range of (0, {}]".format(Constants.attribute_end - Constants.attribute_start)

    attribute_ids = [i for i in range(start_ids, end_ids)]
    return embeddings[attribute_ids]


def get_stop_words_list():
    root_path = os.path.dirname(__file__)   # root_path/misc/utils.corpora.py --> root_path/misc
    root_path = os.path.dirname(root_path)  # root_path/misc --> root_path
    stop_words_list = json.load(open(os.path.join(root_path, 'data', 'en.json'), 'rb'))
    return stop_words_list

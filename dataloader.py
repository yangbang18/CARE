import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import h5py
from config import Constants
import pickle
import os

from misc.utils import(
    resampling,
    get_random_ids_from_the_whole,
    get_random_ids_from_k_snippets,
    get_uniform_ids_from_k_snippets,
    get_uniform_items_from_k_snippets,
    tokenize,
)
from misc.utils_corpora import get_stop_words_list, get_vid2attribute_mappings
from PIL import Image
import glob


def get_frame_ids(n_total_frames, n_frames, random_type):
    if random_type == 'all_random':
        return get_random_ids_from_the_whole(n_total_frames, n_frames)
    elif random_type == 'segment_random':
        return get_random_ids_from_k_snippets(n_total_frames, n_frames)
    elif random_type == 'equally_sampling':
        return get_uniform_ids_from_k_snippets(n_total_frames, n_frames)
    else:
        raise ValueError('We do not support `random_type` = {} now'.format(random_type))


def get_ids_set(mode, split, specific=-1, split_category=None, is_vatex_activate=False):
    if is_vatex_activate:
        for m in ['train', 'validate', 'test']:
            # this is because some videos in the VATEX dataset are not available anymore
            split[m] = split['activate_%s'%m]

    if mode == 'all' and mode not in split:
        split['all'] = split['train'] + split['validate'] + split['test']

    if mode == 'trainval' and mode not in split:
        split['trainval'] = split['train'] + split['validate']
        
    if specific != -1:
        # we only evaluate partial examples with a specific category (MSRVTT, [0, 19])
        ids_set = [int(item) for item in split_category[mode][specific]]
    else:
        # we evaluate all examples regardless of categories
        ids_set = [int(item) for item in split[mode]]
    return ids_set


class VideoOnlyDataset(Dataset):
    def __init__(self, opt, mode, random_type, specific=-1, **kwargs) -> None:
        '''
            the argument `opt` must have the following keys:
                - info_corpus:      str, path to load the preprocessed corpus file
                - modality:         str, specify which modalities to use
                - feats_x:          str or list of str, path(s) to load features of the modality `x`
                - dim_x:            int, the dimension of the features of the modality `x`
                - n_frames:         int, the number of sampled frames for each video
                - load_feats_type:  int, choices = [0, 1] 
            optional (if only if passing the `--with_backbones` argument):
        '''
        Dataset.__init__(self)
        assert mode in ['train', 'validate', 'test', 'all', 'trainval']
        assert random_type in ['segment_random', 'all_random', 'equally_sampling']
        self.opt = opt
        self.mode = mode
        self.random_type = random_type
        
        info = pickle.load(open(opt['info_corpus'], 'rb'))['info']
        self.itoc = info.get('itoc', None)
        self.vid2id = info.get('vid2id', None)

        if opt.get('feats', '') != 'I3D' and opt.get('dataset', 'MSRVTT') == 'VATEX':
            is_vatex_activate = True
        else:
            is_vatex_activate = False

        self.ids_set = get_ids_set(
            mode=mode, 
            split=info['split'], 
            specific=specific, 
            split_category=info.get('split_category', None),
            is_vatex_activate=is_vatex_activate
        )

        self.all_frames_path = opt.get('all_frames_path', '')
        if not self.all_frames_path:
            self.all_frames_path = os.path.join(Constants.base_data_path, opt['dataset'], 'all_frames')
        self.frames_suffix = opt.get('frames_suffix', 'jpg')

        self.has_image_backbone = False
        if 'i' in opt['modality'] \
            and opt.get('with_backbones', []) \
                and opt['with_backbones'][opt['modality'].index('i')].strip():
            assert os.path.exists(self.all_frames_path), self.all_frames_path
            assert 'image_preprocess_func' in kwargs
            self.image_preprocess_func = kwargs['image_preprocess_func']

            self.has_image_backbone = True
            self.is_clip_backbone = 'clip' in opt['with_backbones'][opt['modality'].index('i')].strip()

    def __getitem__(self, index):
        vid = 'video%d' % self.ids_set[index]
        return self._getitem_video_only(vid)

    def __len__(self):
        return len(self.ids_set)
    
    def _getitem_video_only(self, vid):
        if not hasattr(self, 'databases'):
            self.databases = self._make_databases()
        
        return self.get_video_features_by_vid(vid)
    
    def _load_database(self, path):
        if not path: return []
        if not isinstance(path, list): path = [path]
        return [h5py.File(p, 'r') for p in path if '.hdf5' in p]

    def _make_databases(self):
        databases = []
        for char in self.opt['modality'].lower():
            database = self._load_database(self.opt["feats_%s" % char])
            assert len(database) > 0
            databases.append([char, database, self.opt["dim_%s" % char]])
        
        return databases

    def get_database(self, modality):
        if not hasattr(self, 'databases'):
            self.databases = self._make_databases()
        
        for item in self.databases:
            if item[0] == modality:
                return item[1]

    def get_video_features_by_vid(self, vid, load_original_images_to_visualize=False):
        if not hasattr(self, 'databases'):
            self.databases = self._make_databases()

        _dict = {'video_ids': vid}

        if self.opt.get('feats', '') == 'I3D' and self.opt['dataset'] == 'VATEX':
            assert self.vid2id is not None
            vid = self.vid2id[vid] # key_name in I3D.hdf5 is `youtubeid_start_end` rather than `videoXXXX`
        
        frame_ids = get_frame_ids(
            Constants.n_total_frames,
            self.opt['n_frames'], 
            self.random_type
        ) if self.opt['load_feats_type'] == 0 else None

        if frame_ids is not None:
            _dict['frame_ids'] = frame_ids

        _dict['feats'] = []
        for item in self.databases:
            modality = item[0]
            if modality == 'i' and self.has_image_backbone:
                images, *other_info = self._load_images(vid, frame_ids, load_original_images_to_visualize)
                # images will be processed by the image backbone
                _dict['feats'].append(torch.stack(images, dim=0))
                if load_original_images_to_visualize:
                    _dict['original_images'] = other_info[1]
                    _dict['images_basenames'] = other_info[2]
            else:
                if modality == 'r':
                    feats = self.load_r_feats(item, vid)
                elif modality == 't':
                    feats = self.load_t_feats(item, vid)
                else:
                    load_all = True if self.opt['feats'] == 'SwinBERTDense' and modality == 'm' else False
                    feats, *other_info = self._load_feats(item[1:], vid, frame_ids=frame_ids, load_all=load_all)
                    feats = torch.FloatTensor(feats)
                _dict['feats'].append(feats)

            if len(other_info) and self.opt['load_feats_type'] != 0:
                if 'frame_ids' not in _dict:
                    _dict['frame_ids'] = []
                _dict['frame_ids'].append(other_info[0])

        if self.itoc is not None:
            _dict['category'] = torch.LongTensor([self.itoc[int(vid[5:])]])

        return _dict
    
    def load_r_feats(self, item_of_databases, vid):
        raise NotImplementedError()
    
    def load_t_feats(self, item_of_databases, vid):
        raise NotImplementedError()
    
    def get_video_frames_by_vid(self, vid, all_frames_path=''):
        _, frame_ids, original_images, images_basenames = self._load_images(
            vid, load_original_images_to_visualize=True, transform=False, all_frames_path=all_frames_path)
        return original_images
    
    def _load_images(self, vid, frame_ids=None, load_original_images_to_visualize=False, transform=True, all_frames_path=''):
        if not all_frames_path:
            all_frames_path = self.all_frames_path
        frames = sorted(glob.glob(os.path.join(all_frames_path, vid, '*.{}'.format(self.frames_suffix))))
        frames = get_uniform_items_from_k_snippets(frames, Constants.n_total_frames)

        if frame_ids is None:
            frame_ids = get_frame_ids(
                len(frames), 
                self.opt['n_frames'], 
                self.random_type
            )
        
        images, original_images, images_basenames = [], [], []
        for index in frame_ids:
            images_basenames.append(os.path.basename(frames[index]).split('.')[0])

            if transform:
                if self.is_clip_backbone:
                    image = self.image_preprocess_func(Image.open(frames[index]))
                else:
                    image = self.image_preprocess_func(frames[index])
                images.append(image)

            if load_original_images_to_visualize:
                original_images.append(Image.open(frames[index]).convert('RGB'))

        return images, frame_ids, original_images, images_basenames

    def _load_feats(self, data, vid, load_all=False, **kwargs):
        frame_ids = kwargs.get('frame_ids', None)

        databases, dim = data
        max_seq_len = databases[0].get('max_len', self.opt['n_frames'])
        if max_seq_len != self.opt['n_frames']:
            max_seq_len = int(np.asarray(max_seq_len))

        feats = []
        pre_len = None
        for database in databases:
            if vid not in database.keys():
                return np.zeros((max_seq_len, dim)), [_ for _ in range(max_seq_len)]
            else:
                data = np.asarray(database[vid])
                if len(data.shape) == 1:
                    if pre_len is not None:
                        data = data[np.newaxis, :].repeat(pre_len, axis=0)
                    else:
                        data = data[np.newaxis, :].repeat(Constants.n_total_frames, axis=0)
                else:
                    pre_len = data.shape[0]
            feats.append(data)

        if len(feats[0].shape) == 1:
            feats = np.concatenate(feats, axis=0)
            return (feats, )

        feats = np.concatenate(feats, axis=1)
        if load_all:
            return (feats, )

        if self.opt['load_feats_type'] == 0:
            assert frame_ids is not None
        elif self.opt['load_feats_type'] == 1:
            source_length = feats.shape[0]
            if source_length >= self.opt['n_frames']:
                frame_ids = get_frame_ids(
                        source_length, 
                        self.opt['n_frames'], 
                        self.random_type)
            else:
                frame_ids = resampling(source_length, max_seq_len)
        else:
            source_length = feats.shape[0]
            if source_length < max_seq_len:
                frame_ids = resampling(source_length, max_seq_len)
            else:
                frame_ids = [_ for _ in range(feats.shape[0])]

        return feats[frame_ids], frame_ids


class TextOnlyDataset(Dataset):
    def __init__(self, opt, mode, n_caps_per_video, specific=-1, make_infoset=True, **kwargs) -> None:
        '''
            the argument `opt` must have the following keys:
            * simplest version:
                - info_corpus:      str, path to load the preprocessed corpus
                - max_len:          int, maximun length of captions
            * full version:
                - info_corpus:      str, path to load the preprocessed corpus
                - references:       str, path to load references (ground-truth captions w/o preprocessing)
                - max_len:          int, maximun length of captions
                - seed:             int, random seed
                - decoding_type:    str, choices = ['ARFormer', 'NARFormer']
                - visual_word_generation: bool, whether preparing data for the auxiliary task or not
                - beta:             list of two float numbers, specify the lowest and highest masking prob for MLM
                - demand:           list of str, specify which types of words will be treated as visual words,
                                    e.g., ['NOUN', 'VERB']
        '''
        Dataset.__init__(self)
        assert mode in ['train', 'validate', 'test', 'all', 'trainval']
        assert n_caps_per_video >= 0
        self.opt = opt
        self.mode = mode
        self.n_caps_per_video = n_caps_per_video

        data = pickle.load(open(opt['info_corpus'], 'rb'))
        self.captions = data['captions']
        self.pos_tags = data['pos_tags']

        info = data['info']    
        self.itow = info['itow']
        self.wtoi = {w: i for i, w in self.itow.items()}
        self.itoc = pickle.load(open(opt['itoc_path'], 'rb')) if opt.get('itoc_path', '') else info.get('itoc', None)
        self.itop = info.get('itop', None)
        self.vid2id = info.get('vid2id', None)
        self.category_embeddings = info.get('category_embeddings', None)
        self.length_info = info.get('length_info', None)
        self.random = np.random.RandomState(opt.get('seed', 0))
        
        if opt.get('feats', '') != 'I3D' and opt.get('dataset', 'MSRVTT') == 'VATEX':
            is_vatex_activate = True
        else:
            is_vatex_activate = False

        self.ids_set = get_ids_set(
            mode=mode, 
            split=info['split'], 
            specific=specific, 
            split_category=info.get('split_category', None),
            is_vatex_activate=is_vatex_activate
        )

        train_ids_set = get_ids_set(
            mode='train', 
            split=info['split'], 
            specific=specific, 
            split_category=info.get('split_category', None),
            is_vatex_activate=is_vatex_activate
        )
        self.flat_captions = [caption for train_id in train_ids_set for caption in self.captions['video%d'%train_id]]

        self.stop_words_list = get_stop_words_list()
        if make_infoset:
            self.infoset = self._make_infoset()
        
        self.tokenizer = opt.get('tokenizer', None)
        
        self.vid2attr = None
        if data['attribute_flag'] and not self.tokenizer:
            # please run prepare_corpora.py with the `--sort_vocab` and `--attribute_first` arguments
            self.vid2attr = get_vid2attribute_mappings(self.ids_set, self.captions)
    
    def __getitem__(self, index):
        return self._getitem_text_only(index)

    def _getitem_text_only(self, index):
        vid = self.infoset[index]['vid']
        cap_id = self.infoset[index]['cap_id']
        labels = self.infoset[index]['labels']
        taggings = self.infoset[index]['pos_tags']
        
        data = {'video_ids': vid}
        if self.tokenizer is not None:
            data['caption_ids'] = cap_id
            sent = ' '.join([self.itow[i] for i in labels[1:-1]])
            data['input_ids'] = tokenize(self.tokenizer, sent, truncate=True).squeeze()
            data['labels'] = data['input_ids'][1:]
        else:
            data.update(self._prepare_input_ids(cap_id, labels, taggings))

            data['category'] = torch.LongTensor([self.infoset[index]['category']])
            data['category_embs'] = torch.FloatTensor(self.infoset[index]['category_embs'])
            data['length_target'] = torch.FloatTensor(self.infoset[index]['length_target'])
            data['tgt_visual_taggings'] = torch.LongTensor(self._prepare_tgt_visual_taggings(labels, taggings))
            data['non_stop_words_mask'] = torch.LongTensor(self._prepare_non_stop_words_mask(data['labels']))
            data['attribute_mask'] = torch.LongTensor(self._prepare_attribute_mask(data['labels']))
            if self.vid2attr is not None:
                data['labels_attr'] = torch.FloatTensor(self.vid2attr[vid])

        return data
        
    def __len__(self):
        return len(self.infoset)
    
    def _make_infoset(self):
        infoset = []
        self.cap_start_ids = [0]
        self.vid2unique_non_stop_words = {}
        for idx in self.ids_set:
            vid = 'video%d' % idx
            unique_non_stop_words = set()

            category = self.itoc[idx] if self.itoc is not None else 0
            category_embs = self.category_embeddings[category] if self.category_embeddings is not None else [0]
            captions = self.captions[vid]
            pos_tags = self.pos_tags[vid] if self.pos_tags is not None else ([None] * len(captions))
            assert len(captions) == len(pos_tags)

            # prepare length info for each video example, only if decoding_type == 'NARFormmer'
            # e.g., 'video1': [0, 0, 3, 5, 0]
            if self.length_info is None or vid not in self.length_info.keys():
                length_target = np.zeros(self.opt['max_len'])
            else:
                length_target = self.length_info[vid]
                length_target = length_target[:self.opt['max_len']]
                if len(length_target) < self.opt['max_len']:
                    length_target += [0] * (self.opt['max_len'] - len(length_target))

                length_target = np.array(length_target) / sum(length_target)
            
            # decide which captions are used to calculate training/evaluation loss
            if self.n_caps_per_video == 0:
                cap_id_set = [i for i in range(len(captions))]
            elif self.n_caps_per_video == 1 and self.mode != 'train':
                cap_id_set = [0]
            else:
                n_caps_per_video = min(len(captions), self.n_caps_per_video)
                cap_id_set = self.random.choice(
                    [i for i in range(len(captions))], 
                    n_caps_per_video,
                    replace=False
                )
            
            self.cap_start_ids.append(len(cap_id_set))
            
            for cap_id in cap_id_set:
                item = {
                    'vid': vid,
                    'labels': captions[cap_id],
                    'pos_tags': pos_tags[cap_id],
                    'category': category,
                    'category_embs': category_embs,
                    'length_target': length_target,
                    'cap_id': cap_id,
                    }
                infoset.append(item)

                for wid in captions[cap_id][1:-1]:
                    if self.itow[wid] not in self.stop_words_list:
                        unique_non_stop_words.add(wid)
            
            self.vid2unique_non_stop_words[vid] = list(unique_non_stop_words)

        self.cap_start_ids = self.cap_start_ids[:-1]

        if hasattr(self, '_make_infoset_post_processing'):
            infoset = self._make_infoset_post_processing(infoset)
        
        return infoset

    def _prepare_tgt_visual_taggings(self, labels, pos_tagging):
        """
        Get visual tagging from pos_tagging and target sentence.
        Because we want to remove be words from visual words,
        and VERB from pos_tag includes be words, we could not directly use pos_tag instead.
         example:
          sentence        "<bos> a man is watching movie on his phone <eos>"
          visual   tag    [  0   0  1   0    1       1    0  0    1     0 ] with padding
        Notice that <bos> should be remove to match label!
        """
        # remember to exclude <bos> <eos>
        assert self.itop and self.itow

        # sentence is
        # " ".join([self.itow[l] for l in labels])

        # get the position of tokens that have the pos_tag we demand
        visual_word_tag = [0]  # 0 for <bos>
        for i, item in enumerate(pos_tagging[1:-1]):
            w = self.itow[labels[i+1]]
            # we ignore verb ``be''
            if self.itop[item] in ['VERB', 'NOUN'] and w not in ['is', 'are', 'was', 'were', 'be']:
                visual_word_tag.append(1)
            else:
                visual_word_tag.append(0)
        return self._padding(visual_word_tag, add_eos=True)[1:]

    def _prepare_non_stop_words_mask(self, labels):
        '''
            e.g., if a label is     "a man is singing <eos> <pad>",
            then the mask will be   [0, 1, 0,   1,     0,     0]
        '''
        if not hasattr(self, 'stop_words_list'):
            self.stop_words_list = get_stop_words_list()
        
        if isinstance(labels, list):
            # this occurs when `visual_word_generation` is True
            labels = labels[-1]
        
        mask = []
        for label in labels.tolist():
            if label in [Constants.PAD, Constants.EOS]:
                mask.append(0)
            else:
                w = self.itow[label]
                mask.append(0 if w in self.stop_words_list else 1)
        assert len(labels) == len(mask)
        return mask
    
    def _prepare_attribute_mask(self, labels):
        if isinstance(labels, list):
            # this occurs when `visual_word_generation` is True
            labels = labels[-1]

        mask = []
        start = Constants.attribute_start
        if self.opt.get('attribute_prediction_k'):
            end = start + self.opt['attribute_prediction_k']
        else:
            end = Constants.attribute_end

        for label in labels.tolist():
            if start <= label < end:
                mask.append(1)
            else:
                mask.append(0)
        assert len(labels) == len(mask)
        return mask

    def _prepare_input_ids(self, cap_id, ori_labels, taggings):
        _dict = {'caption_ids': cap_id}

        results, info = self._make_source_target(ori_labels, taggings)
        tokens, labels, taggings = map(
            lambda x: results.get(x, None), 
            ["dec_source", "dec_target", "tagging"]
        )

        _dict.update(info)

        if taggings is not None:
            _dict['taggings'] = torch.LongTensor(taggings)
        
        tokens_1 = results.get('dec_source_1', None)
        labels_1 = results.get('dec_target_1', None)
        if tokens_1 is not None:
            assert self.opt.get('visual_word_generation', False) is True
            _dict['input_ids'] = [torch.LongTensor(tokens_1), torch.LongTensor(tokens)]
            _dict['labels'] = [torch.LongTensor(labels_1), torch.LongTensor(labels)]
            
            vmop_crit_flag = self.opt.get('vmop_crit_flag', '')
            if 'S' in vmop_crit_flag or 'I' in vmop_crit_flag:
                if self.opt['decoding_type'] == 'NARFormer':
                    clean_input_ids = self._padding(ori_labels[1:-1], add_eos=False)
                else:
                    clean_input_ids = self._padding(ori_labels, add_eos=True)[:-1]
                _dict['input_ids'].append(torch.LongTensor(clean_input_ids))
        else:
            _dict['input_ids'] = torch.LongTensor(tokens)
            _dict['labels'] = torch.LongTensor(labels)
            _dict['labels'] = torch.LongTensor(labels)

        return _dict

    def _make_source_target(self, target, tagging):
        results, info = {}, {}

        if self.opt.get('decoding_type', 'ARFormer') == 'NARFormer':
            results = self._source_target_mlm(target[1:-1]) # exclude <bos> <eos>
        else:
            results = {
                'dec_source': self._padding(target, add_eos=True), 
                'dec_target': self._padding(target, add_eos=True)
            }
        
            results['dec_source'] = results['dec_source'][:-1]
            results['dec_target'] = results['dec_target'][1:]

        assert len(results['dec_source']) == len(results['dec_target'])

        if self.opt.get('visual_word_generation', False):
            results.update(self._source_target_visual_word(target=target, pos_tag=tagging))

        if 'tagging' not in results.keys() and tagging is not None:
            results['tagging'] = self._padding(tagging, add_eos=True)

        return results, info

    def _source_target_mlm(self, target):
        assert target[0] != Constants.BOS
        assert target[-1] != Constants.EOS

        beta_low, beta_high = self.opt.get('beta', [0, 1])

        min_num_masks = 1
        dec_source = torch.LongTensor(target)
        dec_target_cp = torch.LongTensor(target)
        dec_target = torch.LongTensor([Constants.PAD] * len(dec_source))

        if self.mode == 'train':
            if min_num_masks >= len(dec_source):
                ind = np.array([],dtype=np.uint8)
            else:
                low = max(int(len(dec_source) * beta_low), min_num_masks)
                high = max(int(len(dec_source) * beta_high), min_num_masks)
                if high == low:
                    high += 1
                sample_size = self.random.randint(low, high)
                ind = self.random.choice(len(dec_source) , size=sample_size, replace=False)
            
            if len(ind):
                dec_source[ind] = Constants.MASK
                dec_target[ind] = dec_target_cp[ind]
        else:
            dec_source[dec_source!=Constants.PAD] = Constants.MASK
            dec_target = dec_target_cp           

        dec_source = self._padding(dec_source.tolist(), add_eos=False)
        dec_target = self._padding(dec_target.tolist(), add_eos=False)
        
        return {'dec_source': dec_source, 'dec_target': dec_target}
    
    def _source_target_visual_word(self, **kwargs):
        target = kwargs['target']
        pos_tag = kwargs['pos_tag']
        sent_length = len(target[1:-1]) # exclude <bos> <eos>

        visual_tag = Constants.VIS
        target_tag = Constants.MASK

        if self.mode != 'train':
            dec_target_1 = [0]
            dec_source_1 = [0]
        else:
            assert len(target) == len(pos_tag)
            assert self.itop is not None

            dec_source_1 = self._padding(
                [visual_tag] * (sent_length if self.opt['decoding_type'] == 'NARFormer' else len(target)), 
                add_eos=False if self.opt['decoding_type'] == 'NARFormer' else True
            )

            # get the position of tokens that have the pos_tag we demand
            pos_satisfied_ind = []
            for i, item in enumerate(pos_tag[1:-1]):
                w = self.itow[target[i+1]]
                # we ignore verb ``be''
                if self.itop[item] in self.opt['demand'] and w not in ['is', 'are', 'was', 'were', 'be']:
                    pos_satisfied_ind.append(i)

            pos_satisfied_ind = np.array(pos_satisfied_ind)
            
            # decoder1 need to predict tokens with satisfied pos_tag from scratch
            # meanwhile, decoder1 should learn to keep the remaining tokens (i.e., <mask>) unchanged
            dec_target_1 = torch.LongTensor([target_tag] * sent_length)
            dec_target_cp = torch.LongTensor(target[1:-1])
            dec_target_1[pos_satisfied_ind] = dec_target_cp[pos_satisfied_ind]

            if self.opt['decoding_type'] == 'NARFormer':
                dec_target_1 = self._padding(dec_target_1.tolist(), add_eos=False)
            else:
                # when training with autoregressive transformer, the first token will be ignored, i.e., label = dec_target_1[1:]
                dec_target_1 = self._padding([target[0]] + dec_target_1.tolist() + [Constants.EOS], add_eos=True)

        return {'dec_source_1': dec_source_1, 'dec_target_1': dec_target_1}

    def _padding(self, seq, add_eos=True, max_len=None, padding_token_id=Constants.PAD):
        if seq is None:
            return None
        
        if max_len is None:
            max_len = self.opt['max_len']

        res = seq.copy()
        if len(res) > max_len:
            res = res[:max_len]
            if add_eos:
                res[-1] = Constants.EOS
        else:
            res += [padding_token_id] * (max_len - len(res))
        return res
    
    def get_references(self):
        if getattr(self, 'references', None) is None:
            self.references = pickle.load(open(self.opt['reference'], 'rb'))
        return self.references

    def get_preprocessed_references(self):
        return self.captions

    def get_gt_sentences_by_vid(self, vid):
        if getattr(self, 'references', None) is None:
            self.references = pickle.load(open(self.opt['reference'], 'rb'))
        return [item['caption'] for item in self.references[vid]]
    
    def get_preprocessed_gt_sentences_by_vid(self, vid, add_special_tokens=False):
        if add_special_tokens:
            return [" ".join([self.itow[wid] for wid in item]) for item in self.captions[vid]]
        else:
            return [" ".join([self.itow[wid] for wid in item[1:-1]]) for item in self.captions[vid]]

    def get_vocab_size(self):
        return len(self.get_vocab())

    def get_vocab(self):
        return self.itow
    
    def preprocess_space_separated_text(self, text, add_special_tokens=True):
        if not isinstance(text, list):
            text = text.split(' ')
        label = [self.wtoi[w] for w in text]
        if add_special_tokens:
            label = [Constants.BOS] + label + [Constants.EOS]
        return label


class JointDataset(VideoOnlyDataset, TextOnlyDataset):
    def __init__(self, opt, mode, print_info=False, specific=-1, **kwargs):
        if mode != 'train' or kwargs.get('is_validation', False):
            random_type = 'equally_sampling'
            n_caps_per_video = 1 if not kwargs.get('all_caps', False) else 0
        else:
            random_type = opt.get('random_type', 'segment_random')
            n_caps_per_video = opt.get('n_caps_per_video', 0)

        VideoOnlyDataset.__init__(self, opt, mode, random_type, specific, **kwargs)
        TextOnlyDataset.__init__(self, opt, mode, n_caps_per_video, specific, **kwargs)
        
        if print_info:
            self.print_info()
    
    def print_info(self):
        print('Dataset Information:')
        print('- the number of videos in the set `{}`: {}'.format(
            self.mode, len(self.ids_set))
        )
        print('- the number of samples (n_caps_per_video={}): {}'.format(
            self.n_caps_per_video, len(self.infoset))
        )
        print('- vocab size is', len(self.itow))
        print('- the maximum sequence length (max_len) is set to', self.opt['max_len'])
        
        print('Modality Information:')
        for char in self.opt['modality'].lower():
            print('- loading feats_{} ({}) from {}'.format(
                char, self.opt['dim_' + char], self.opt['feats_' + char])
            )
        print('- load feats type: %d' % self.opt['load_feats_type'])
        print('- the number of sampled frames is set to', self.opt['n_frames'])

    def get_specific_data_by_vid_and_cap_id(self, vid, cap_id=None, device='cpu', text=None, load_original_images_to_visualize=False):
        assert cap_id is not None or text is not None
        data = self.get_video_features_by_vid(vid, load_original_images_to_visualize)

        if text is not None:
            label = self.preprocess_space_separated_text(text)
            tagging = None
            cap_id = -1
        else:
            label = self.captions[vid][cap_id]
            tagging = self.pos_tags[vid][cap_id]
        data.update(self._prepare_input_ids(cap_id, label, tagging))
        
        data['non_stop_words_mask'] = torch.LongTensor(self._prepare_non_stop_words_mask(data['labels']))

        if self.vid2attr is not None:
            data['labels_attr'] = torch.FloatTensor(self.vid2attr[vid])

        for k in data.keys():
            if k not in ['frame_ids', 'video_ids', 'caption_ids', 'original_images', 'images_basenames']:
                if isinstance(data[k], list):
                    data[k] = [item.unsqueeze(0).to(device) for item in data[k]]
                else:
                    data[k] = data[k].unsqueeze(0)
                    data[k] = data[k].to(device)

        data['video_ids'] = [data['video_ids']] # batch size = 1, ensure no errors occur when calling model.translate_step
        return data

    def _make_infoset_post_processing(self, infoset):
        return infoset

    def __getitem__(self, index):
        vid = self.infoset[index]['vid']
        
        data = {}
        data.update(self._getitem_video_only(vid))
        data.update(self._getitem_text_only(index))
        
        if 'rnn' in self.opt.get('decoder', '').lower():
            # if a video has the category of 1, 
            # then [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, ...] is treated as category features
            category_one_hot = [0] * self.opt.get('num_category', 20)
            category_one_hot[self.infoset[index]['category']] = 1
            data['category'] = torch.FloatTensor(category_one_hot)

        if 'clip_scores' in self.infoset[index]:
            if self.opt['load_feats_type'] == 0:
                frame_ids = data['frame_ids']
            else:
                assert 'i' in self.opt['modality']
                index_of_the_image_modality = self.opt['modality'].index('i')
                frame_ids = data['frame_ids'][index_of_the_image_modality]
            
            clip_scores = self.infoset[index]['clip_scores']
            clip_scores = clip_scores[:self.opt['max_len']-1, frame_ids]
            data['clip_scores'] = torch.FloatTensor(clip_scores)
        
        return data

    def __len__(self):
        return len(self.infoset)
    
    def load_r_feats(self, item_of_databases, vid):
        # get embeddings of topk captions related to vid
        db = item_of_databases[1][0]
        feats = np.asarray(db[vid])
        feats = feats[:self.opt['retrieval_topk'], :]
        feats = torch.FloatTensor(feats)   
        return feats
    
    def load_t_feats(self, item_of_databases, vid):
        # get topk captions related to vid
        db = item_of_databases[1][0]
        captions = self.get_retrieval_captions(vid, db=db)
        feats = torch.LongTensor([
            self._padding(cap[1:-1] if self.opt.get('exclude_eos', False) else cap[1:], add_eos=False)
            for cap in captions
        ])
        return feats

    def get_retrieval_captions(self, vid, db, topk=None):
        indices = np.asarray(db[vid + '_i'])
        topk = topk or self.opt['retrieval_topk']
        indices = indices[:topk]

        captions = []
        for ind in indices:
            captions.append(self.flat_captions[ind])
        
        return captions


def get_loader(opt, mode, print_info=False, specific=-1, **kwargs):
    dataset_type = kwargs.get('dataset_type', 'joint')
    if dataset_type == 'video':
        dataset_class = VideoOnlyDataset
    elif dataset_type == 'text':
        dataset_class = TextOnlyDataset
    else:
        dataset_class = JointDataset

    dataset = dataset_class(opt, mode, print_info=print_info, specific=specific, **kwargs)
    batch_size = kwargs.get('batch_size', opt.get('batch_size', 64))
    
    if kwargs.get('all_samples_one_batch', False):
        batch_size = len(dataset)

    not_shuffle = kwargs.get('not_shuffle', False)
    num_workers = kwargs.get('num_workers', opt.get("num_workers", 0))

    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True if (mode=='train' and not not_shuffle) else False,
        num_workers=num_workers,
    )

import torch
import numpy as np
import random
import os
from config import Constants
import pandas
import json
from typing import Union, List
import yaml


def load_yaml(args, key, yaml_path=None, yaml_data=None, modify_scope=False, name_to_path=False):
    if not key or key is None:
        return None

    assert yaml_path is not None or yaml_data is not None
    if yaml_data is None:
        yaml_data = yaml.full_load(open(yaml_path))
    
    assert key in yaml_data.keys(), f"`{key}` can not be found in {yaml_path}"

    specific_data = yaml_data[key]

    if 'inherit_from' in specific_data.keys():
        inherit_from = specific_data.pop('inherit_from')
        if isinstance(inherit_from, list):
            for new_key in inherit_from:
                load_yaml(args, key=new_key, yaml_path=yaml_path, yaml_data=yaml_data, name_to_path=name_to_path)    
        else:
            load_yaml(args, key=inherit_from, yaml_path=yaml_path, yaml_data=yaml_data, name_to_path=name_to_path)

    new_scope = key
    format_str = None
    if modify_scope:
        if 'scope_format' in specific_data:
            format_str, names = specific_data.pop('scope_format')
        elif hasattr(args, 'scope_format'):
            format_str, names = args.scope_format
            del args.scope_format

    for k, v in specific_data.items():
        if name_to_path and 'name' in k:
            path_k = k.replace('name', 'path')
            print(path_k, v)
            setattr(args, path_k, os.path.join(args.base_data_path if args.base_data_path else Constants.base_data_path, args.dataset, v))
        else:
            setattr(args, k, v)
        
    if modify_scope:
        values = []
        if format_str is not None:
            for name in names:
                this_value = getattr(args, name)
                if isinstance(this_value, list):
                    values.append('-'.join([str(item) for item in this_value]))
                else:
                    values.append(this_value)
            new_scope = format_str.format(*values)
        args.scope = new_scope + '_' + args.scope if args.scope else new_scope


def check_whether_to_load_weights(args):
    if args.task:
        yaml_data = yaml.full_load(open('./config/tasks.yaml'))
        specific_data = yaml_data[args.task]

        if specific_data.get('weights_from_inherit', False):
            assert 'inherit_from' in specific_data.keys(), specific_data.keys()

            def get_scope_format(yaml_data, key):
                if isinstance(key, list):
                    key = key[0]
                if 'scope_format' in yaml_data[key]:
                    return yaml_data[key]['scope_format']
                assert 'inherit_from' in yaml_data[key], "{}: {}".format(key, yaml_data[key].keys())
                return get_scope_format(yaml_data, yaml_data[key]['inherit_from'])

            format_str, names = get_scope_format(yaml_data, specific_data['inherit_from'])

            values = []
            for name in names:
                this_value = getattr(args, name)
                if isinstance(this_value, list):
                    values.append('-'.join([str(item) for item in this_value]))
                else:
                    values.append(this_value)
            
            inherit_scope = format_str.format(*values)
            #inherit_scope = inherit_scope + '_' + args.scope if args.scope else inherit_scope

            args.load_model_weights_from = os.path.join(
                Constants.base_checkpoint_path,
                args.dataset,
                args.method,
                specific_data['inherit_from'],
                inherit_scope,
                'best.ckpt',
            )


def get_shape_and_device(tensor):
    if isinstance(tensor, list) or isinstance(tensor, tuple):
        return get_shape_and_device(tensor[0])
    return tensor.shape, tensor.device


def to_device(
        tensor: Union[torch.Tensor, List[torch.Tensor]], 
        device: torch.device
    ) -> Union[torch.Tensor, List[torch.Tensor]]:

    if isinstance(tensor, list):
        return [to_device(item, device) for item in tensor]
    return tensor.to(device)


def to_sentence(hyp, vocab, break_words=[Constants.EOS, Constants.PAD], skip_words=[], extra_mappings={}, add_eos=False):
    sent = []
    if len(extra_mappings):
        new_vocab = {**vocab, **extra_mappings}
    else:
        new_vocab = vocab

    flag = False
    for word_id in hyp:
        if flag:
            break
        if word_id in skip_words:
            continue
        if word_id in break_words:
            if add_eos and word_id == Constants.EOS:
                flag = True
            else:
                break
        word = new_vocab[word_id]
        sent.append(word)
    return ' '.join(sent)


def to_sentence_with_tokenizer(hyp, tokenizer):
    i = 0
    while i < len(hyp):
        if hyp[i] == Constants.EOS:
            break
        i = i + 1
    
    hyp = hyp[:i]
    sent = tokenizer.decode(hyp).strip()
    return sent
    

def remove_repeat_n_grame(sent, n):
    length = len(sent)
    rec = {}
    result_sent = []
    for i in range(length-n+1):
        key = ' '.join(sent[i:i+n])
        if key in rec.keys():
            dis = i - rec[key] - n
            if dis in [0,1]:
                result_sent += sent[:i-dis]
                if i+n <length:
                    result_sent += sent[i+n:]
                return result_sent, False
        else:
            rec[key] = i
    return sent, True


def duplicate(sent):
    sent = sent.split(' ')
    res = {}
    for i in range(4, 0, -1):
        jud = False
        while not jud:
            sent, jud = remove_repeat_n_grame(sent, i)
            if not jud:
                res[i] = res.get(i, 0) + 1
            else:
                break
    res_str = []
    for i in range(1, 5):
        res_str.append('%d-gram: %d' % (i, res.get(i, 0)))
    return ' '.join(sent), '\t'.join(res_str)


def cal_gt_n_gram(data, vocab, splits, n=1):
    gram_count = {}
    gt_sents = {}
    for i in splits['train']:
        k = 'video%d'% int(i)
        caps = data[k]
        for tmp in caps:
            cap = [vocab[wid] for wid in tmp[1:-1]]
            gt_sents[' '.join(cap)] = gt_sents.get(' '.join(cap), 0) + 1
            for j in range(len(cap)-n+1):
                key = ' '.join(cap[j:j+n])
                gram_count[key] = gram_count.get(key, 0) + 1
    return gram_count, gt_sents


def cal_n_gram(data, n=1):
    gram_count = {}
    sents = {}
    ave_length, count = 0, 0
    for k in data.keys():
        for i in range(len(data[k])):
            sents[data[k][i]['caption']] = sents.get(data[k][i]['caption'], 0) + 1
            cap = data[k][i]['caption'].split(' ')
            ave_length += len(cap)
            count += 1
            for j in range(len(cap)-n+1):
                key = ' '.join(cap[j:j+n])
                gram_count[key] = gram_count.get(key, 0) + 1
    return gram_count, sents, ave_length/count, count


def analyze_length_novel_unique(gt_data, data, vocab, splits, n=1, calculate_novel=True):
    novel_count = 0
    hy_res, hy_sents, ave_length, hy_count = cal_n_gram(data, n)
    if calculate_novel:
        gt_res, gt_sents = cal_gt_n_gram(gt_data, vocab, splits, n)
        for k1 in hy_sents.keys():
            if k1 not in gt_sents.keys():
                novel_count += 1

    novel = novel_count / hy_count
    unique = len(hy_sents.keys()) / hy_count
    vocabulary_usage = len(hy_res.keys())

    gram4, _, _, _ = cal_n_gram(data, n=4)
    return ave_length, novel, unique, vocabulary_usage, hy_res, len(gram4)


def get_words_with_specified_tags(word_to_ix, seq, index_set, demand=['NOUN', 'VERB'], ignore_words=['is', 'are', '<mask>']):
    import nltk
    assert isinstance(index_set, set)
    res = nltk.pos_tag(seq.split(' '))
    for w, t in res:
        if Constants.pos_tag_mapping[t] in demand and w not in ignore_words:
            index_set.add(word_to_ix[w])


def enlarge(info, beam_size, bsz=None):
    if bsz is not None and info.shape[0] != bsz:
        assert info.shape[0] == 1
        return info

    bsz, *rest_shape = info.shape
    if len(rest_shape) == 3:
        info = info.unsqueeze(1).repeat(1, beam_size, 1, 1, 1)
    elif len(rest_shape) == 2:
        info = info.unsqueeze(1).repeat(1, beam_size, 1, 1)
    elif len(rest_shape) == 1:
        info = info.unsqueeze(1).repeat(1, beam_size, 1)
    else:
        info = info.unsqueeze(1).repeat(1, beam_size)
    return info.contiguous().view(bsz * beam_size, *rest_shape)


def auto_enlarge(info, beam_size, bsz=None):
    if isinstance(info, dict):
        return {
            key: auto_enlarge(info[key], beam_size, bsz)
            for key in info.keys()
        }
    elif isinstance(info, list):
        if isinstance(info[0], tuple):
            return [
                tuple([enlarge(_, beam_size, bsz) for _ in item])
                for item in info
            ]
        else:
            return [enlarge(item, beam_size, bsz) for item in info]
    else:
        if isinstance(info, tuple):
            return tuple([enlarge(item, beam_size, bsz) for item in info])
        else:
            return enlarge(info, beam_size, bsz)


def filter_weight_decay(model, weight_decay=1e-5, filter_biases=False, skip_list=(), skip_substr_list=()):
    def is_substr_in(name):
        for substr in skip_substr_list:
            if substr in name:
                return True
        return False

    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if filter_biases and param.dim() == 1:
            print(f'weight decay is not applied to the parameter `{name}`')
            no_decay.append(param)
        elif name in skip_list or is_substr_in(name):
            print(f'weight decay is not applied to the parameter `{name}`')
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def resampling(source_length, target_length):
    return [round(i * (source_length-1) / (target_length-1)) for i in range(target_length)]


def get_uniform_ids_from_k_snippets(length, k, offset=0):
    uniform_ids = []
    bound = [int(i) for i in np.linspace(0, length, k+1)]
    for i in range(k):
        idx = (bound[i] + bound[i+1]) // 2
        uniform_ids.append(idx + offset)
    return uniform_ids


def get_random_ids_from_k_snippets(length, k, offset=0):
    random_ids = []
    bound = [int(i) for i in np.linspace(0, length, k+1)]
    for i in range(k):
        idx = np.random.randint(bound[i], bound[i+1])
        random_ids.append(idx + offset)
    return random_ids


def get_random_ids_from_the_whole(length, k, offset=0):
    random_ids = random.sample([i for i in range(length)], k)
    random_ids = [i + offset for i in random_ids]
    return sorted(random_ids)


def get_uniform_items_from_k_snippets(items, k):
    uniform_ids = get_uniform_ids_from_k_snippets(len(items), k)
    return [items[idx] for idx in uniform_ids]


def get_ids_of_keyframes(total_frames_of_a_video, k, identical=True, offset=0):
    if identical:
        ''' In our implementation, we follow two steps:
            1. extract 60 features to represent a video (see the `hdf5` feature files);
            2. feed uniformly-sampled k features into the captioning model during inference.
        '''
        assert k < 60
        uniform_ids = get_uniform_ids_from_k_snippets(total_frames_of_a_video, Constants.n_total_frames) # step1
        real_ids = get_uniform_items_from_k_snippets(uniform_ids, k) # step2
    else:
        ''' the real_ids is slightly different from the one above
            e.g., with total_frames_of_a_video = 198 and k = 8,
            identical = True:  real_ids = [11, 37, 60, 87, 110, 136, 159, 186]
            identical = False: real_ids = [12, 36, 61, 86, 111, 135, 160, 185]
        '''
        real_ids = get_uniform_ids_from_k_snippets(total_frames_of_a_video, k)

    if offset:
        real_ids = [idx + offset for idx in real_ids]

    return real_ids


def save_dict_to_csv(path, file_name, dict_data):
    os.makedirs(path, exist_ok=True)
    if ".csv" not in file_name:
        file_name = file_name + ".csv"
    csv_path = os.path.join(path, file_name)
    df_scores = pandas.DataFrame([dict_data])
    if os.path.exists(csv_path):
        df_scores.to_csv(csv_path, index=False, mode='a', header=False)
    else:
        df_scores.to_csv(csv_path, index=False, mode='w')


def cal_gt_n_gram(data, vocab, splits, n=1):
    gram_count = {}
    gt_sents = {}
    for i in splits['train']:
        k = 'video%d'% int(i)
        caps = data[k]
        for tmp in caps:
            cap = [vocab[wid] for wid in tmp[1:-1]]
            gt_sents[' '.join(cap)] = gt_sents.get(' '.join(cap), 0) + 1
            for j in range(len(cap)-n+1):
                key = ' '.join(cap[j:j+n])
                gram_count[key] = gram_count.get(key, 0) + 1
    return gram_count, gt_sents


def cal_n_gram(data, n=1):
    gram_count = {}
    sents = {}
    ave_length, count = 0, 0
    for k in data.keys():
        for i in range(len(data[k])):
            sents[data[k][i]['caption']] = sents.get(data[k][i]['caption'], 0) + 1
            cap = data[k][i]['caption'].split(' ')
            ave_length += len(cap)
            count += 1
            for j in range(len(cap)-n+1):
                key = ' '.join(cap[j:j+n])
                gram_count[key] = gram_count.get(key, 0) + 1
    return gram_count, sents, ave_length/count, count


def analyze_length_novel_unique(gt_data, data, vocab, splits, n=1):
    hy_res, hy_sents, ave_length, hy_count = cal_n_gram(data, n)
    
    novel_count = 0
    _, gt_sents = cal_gt_n_gram(gt_data, vocab, splits, n)
    for k1 in hy_sents.keys():
        if k1 not in gt_sents.keys():
            novel_count += 1

    novel = novel_count / hy_count
    unique = len(hy_sents.keys()) / hy_count
    vocabulary_usage = len(hy_res.keys())

    return ave_length, novel, unique, vocabulary_usage


def tokenize(tokenizer, texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result

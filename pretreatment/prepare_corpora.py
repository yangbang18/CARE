import sys
sys.path.append('..')
sys.path.append('.')
import argparse
from config import Constants
import os
import pickle
from misc import utils_corpora

# only words that occur more than this number of times will be put in vocab
word_count_threshold = {
    'MSVD': 2,
    'MSRVTT': 2,
    'VATEX': 2,
}


def main(args):
    func_name = 'preprocess_%s' % args.dataset
    preprocess_func = getattr(utils_corpora, func_name, None)
    if preprocess_func is None:
        raise ValueError('We can not find the function %s in misc/utils_corpora.py' % func_name)

    results = preprocess_func(args.base_pth)
    split = results['split']
    raw_caps_train = results['raw_caps_train']
    raw_caps_all = results['raw_caps_all']
    references = results.get('references', None)

    vid2id = results.get('vid2id', None)
    itoc = results.get('itoc', None)
    split_category = results.get('split_category', None)
    
    # create the vocab
    vocab = utils_corpora.build_vocab(
        raw_caps_train, 
        word_count_threshold[args.dataset],
        sort_vocab=args.sort_vocab,
        attribute_first=args.attribute_first
        )
    itow, captions, itop, pos_tags = utils_corpora.get_captions_and_pos_tags(raw_caps_all, vocab)

    length_info = utils_corpora.get_length_info(captions)
    #next_info = get_next_info(captions, split)

    info = {
        'split': split,                # {'train': [0, 1, 2, ...], 'validate': [...], 'test': [...]}
        'vid2id': vid2id,
        'split_category': split_category,
        'itoc': itoc,
        'itow': itow,                       # id to word
        'itop': itop,                       # id to POS tag
        'length_info': length_info,         # id to length info
    }

    if args.pretrained_path:
        utils_corpora.prepare_pretrained_word_embeddings(args, itow)
        if itoc is not None:
            info['category_embeddings'] = utils_corpora.prepare_category_embeddings(args)

    pickle.dump({
            'info': info,
            'captions': captions,
            'pos_tags': pos_tags,
            'attribute_flag': args.sort_vocab and args.attribute_first,
            # when `attribute_flag` is True, it enables the dataloader 
            # to easily obtain multi-hot attribute labels for a given video
        }, 
        open(args.corpus, 'wb')
    )

    if references is not None:
        pickle.dump(
            references,
            open(args.refs, 'wb')
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='MSRVTT', type=str, choices=['MSVD', 'MSRVTT', 'VATEX'])
    parser.add_argument('-sort', '--sort_vocab', default=False, action='store_true')
    parser.add_argument('-attr', '--attribute_first', default=False, action='store_true')

    parser.add_argument('-pp', '--pretrained_path', default='', type=str, 
            help='path of the file that stores pretrained word embeddings (e.g., glove.840B.300d.txt); '
            'if specified, pretrained word embeddings of the given dataset will be extracted and stored.')
    parser.add_argument('-pd', '--pretrained_dim', default=300, type=int, 
            help='dimension of the pretrained word embeddings')
    parser.add_argument('-sn', '--save_name', default='embs.npy', type=str, 
            help='the filename to save pretrained word embeddings of the given datasets')
    
    parser.add_argument('--base_data_path', type=str)
                
    args = parser.parse_args()
    
    assert args.dataset in word_count_threshold.keys()
    
    args.base_pth = os.path.join(
        Constants.base_data_path if args.base_data_path is None else args.base_data_path, args.dataset)
    args.corpus = os.path.join(args.base_pth, 'info_corpus.pkl')
    args.refs = os.path.join(args.base_pth, 'refs.pkl')

    print(__file__)
    main(args)

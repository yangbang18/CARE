import os
import json
import pickle
import argparse
from config import Constants
from misc.cocoeval import COCOScorer, suppress_stdout_stderr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='translate.py')
    parser.add_argument('json_path', type=str)
    parser.add_argument("--dataset", type=str, default='MSRVTT', choices=['MSVD', 'MSRVTT', 'VATEX'])
    parser.add_argument("--base_data_path", type=str)
    args = parser.parse_args()

    base_data_path = args.base_data_path if args.base_data_path is not None else Constants.base_data_path
    preds = json.load(open(args.json_path, 'rb'))

    ref_path = os.path.join(base_data_path, args.dataset, 'refs.pkl')
    print('Loading references from', ref_path)
    references = pickle.load(open(ref_path, 'rb'))

    scorer = COCOScorer()
    with suppress_stdout_stderr():
        scores, detail_scores = scorer.score(references, preds, preds.keys())

    print(scores)
    

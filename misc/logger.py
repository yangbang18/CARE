import csv
import os
import shutil
import json
import numpy as np
from queue import PriorityQueue
from tqdm import tqdm
import torch

class CsvLogger:
    def __init__(self, filepath='./', filename='validate_record.csv', data=None, fieldsnames=['epoch', 'train_loss', 'val_loss', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr']):
        self.log_path = filepath
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        if filename:
            self.log_name = filename
            self.csv_path = os.path.join(self.log_path, self.log_name)
            self.fieldsnames = fieldsnames

            if not os.path.exists(self.csv_path):
                with open(self.csv_path, 'w') as f:
                    writer = csv.DictWriter(f, fieldnames=self.fieldsnames)
                    writer.writeheader()

            self.data = {}
            for field in self.fieldsnames:
                self.data[field] = []
            if data is not None:
                for d in data:
                    d_num = {}
                    for key in d:
                        d_num[key] = float(d[key]) if key != 'epoch' else int(d[key])
                    self.write(d_num)

    def write(self, data):
        for k in self.data:
            self.data[k].append(data[k])
        data = {k:v for k, v in data.items() if k in self.data.keys()}
        with open(self.csv_path, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldsnames)
            writer.writerow(data)

    def write_text(self, text, print_t=True):
        with open(os.path.join(self.log_path, 'log.txt'), 'a') as f:
            f.write('{}\n'.format(text))
        if print_t:
            tqdm.write(text)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1, multiply=True):
        self.val = val
        if multiply:
            self.sum += val * n
        else:
            self.sum += val
        self.count += n
        self.avg = self.sum / self.count

class ModelNode(object):
    def __init__(self, res, model_path, key='Sum'):
        self.res = res
        self.model_path = model_path
        self.key = key

    def __lt__(self, other): 
        return self.res[self.key] < other.res[self.key]   

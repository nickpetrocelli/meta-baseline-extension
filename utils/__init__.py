import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
import cvxpy as cp
import math

from . import few_shot, robust_mean_pgd


_log_path = None

def set_log_path(path):
    global _log_path
    _log_path = path

def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def set_gpu(gpu):
    print('set gpu:', gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                or input('{} exists, remove? ([y]/n): '.format(path)) != 'n'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def time_str(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    if t >= 60:
        return '{:.1f}m'.format(t / 60)
    return '{:.1f}s'.format(t)

def robust_proto_pgd(x_shot, epsilon=0.2):
    # robust mean estimation using projected gradient descent
    # assumes using l2 norm/sqr distance
    # x_shot: [num_episodes x n_way x n_shot x embedding_dim]
    #print(f"x shot size: {x_shot.size()}")

    # convert x_shot to numpy 
    x_shot_arr = x_shot.detach().cpu().numpy()

    x_shot_arr_shape = np.shape(x_shot_arr)
    results = []
    for ep in range(x_shot_arr_shape[0]):
        ep_results = []
        for way in range(x_shot_arr_shape[1]):
            ep_results.append(robust_mean_pgd.robust_mean_pgd(x_shot_arr[ep][way], epsilon))
        results.append(ep_results)

    results_arr = np.array(results)
    # TODO parallel?
    # currently assuming one gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results_ten = torch.tensor(results, device=device)
    #print(f"results_ten size: {results_ten.size()}")
    return results_ten


    # get medians
    # x_shot_medians = torch.median(x_shot, dim=-2).values.detach().cpu().numpy
    # print(f'medians shape: {x_shot_medians.shape}')

    # d = x_shot.size()[-1] # this should always be true?

    # # for each way
    # protos = []
    # for c in range(x_shot_medians.shape[0]):
    #     # should be able to derive from shape
    #     N = 0 # TODO

    #     v = x_shot_medians[c]
    #     # for O(log d) iters
    #     for _ in range(math.ceil(math.log(x_shot_medians.shape[1]))):
    #         assert False, "broken"


    

def compute_logits(feat, proto, metric='dot', temp=1.0):
    assert feat.dim() == proto.dim()
    #checking internal dimension
    #print(f'proto size: {proto.size()}')

    if feat.dim() == 2:
        if metric == 'dot':
            logits = torch.mm(feat, proto.t())
        elif metric == 'cos':
            logits = torch.mm(F.normalize(feat, dim=-1),
                              F.normalize(proto, dim=-1).t())
        elif metric == 'sqr':
            logits = -(feat.unsqueeze(1) -
                       proto.unsqueeze(0)).pow(2).sum(dim=-1)

    elif feat.dim() == 3:
        if metric == 'dot':
            logits = torch.bmm(feat, proto.permute(0, 2, 1))
        elif metric == 'cos':
            logits = torch.bmm(F.normalize(feat, dim=-1),
                               F.normalize(proto, dim=-1).permute(0, 2, 1))
        elif metric == 'sqr':
            logits = -(feat.unsqueeze(2) -
                       proto.unsqueeze(1)).pow(2).sum(dim=-1)

    return logits * temp


def compute_acc(logits, label, reduction='mean'):
    ret = (torch.argmax(logits, dim=1) == label).float()
    if reduction == 'none':
        return ret.detach()
    elif reduction == 'mean':
        return ret.mean().item()


def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(params, name, lr, weight_decay=None, milestones=None):
    if weight_decay is None:
        weight_decay = 0.
    if name == 'sgd':
        optimizer = SGD(params, lr, momentum=0.9, weight_decay=weight_decay)
    elif name == 'adam':
        optimizer = Adam(params, lr, weight_decay=weight_decay)
    if milestones:
        lr_scheduler = MultiStepLR(optimizer, milestones)
    else:
        lr_scheduler = None
    return optimizer, lr_scheduler


def visualize_dataset(dataset, name, writer, n_samples=16):
    demo = []
    for i in np.random.choice(len(dataset), n_samples):
        demo.append(dataset.convert_raw(dataset[i][0]))
    writer.add_images('visualize_' + name, torch.stack(demo))
    writer.flush()


def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


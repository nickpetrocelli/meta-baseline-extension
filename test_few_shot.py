import argparse
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import datasets
import models
import utils
import utils.few_shot as fs
from datasets.samplers import CategoriesSampler
import random
import math


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return h


def main(config, args):
    # dataset
    dataset = datasets.make(config['dataset'], **config['dataset_args'])
    utils.log('dataset: {} (x{}), {}'.format(
            dataset[0][0].shape, len(dataset), dataset.n_classes))
    if not args.sauc:
        n_way = 5
    else:
        n_way = 2
    n_shot, n_query = args.shot, 15
    n_batch = 200
    ep_per_batch = 4
    batch_sampler = CategoriesSampler(
            dataset.label, n_batch, n_way, n_shot + n_query,
            ep_per_batch=ep_per_batch)
    loader = DataLoader(dataset, batch_sampler=batch_sampler,
                        num_workers=8, pin_memory=True)

    # model
    if config.get('load') is None:
        model = models.make('meta-baseline', encoder=None)
    else:
        model = models.load(torch.load(config['load']))

    if config.get('load_encoder') is not None:
        encoder = models.load(torch.load(config['load_encoder'])).encoder
        model.encoder = encoder

    if config.get('_parallel'):
        model = nn.DataParallel(model)

    model.eval()
    utils.log('num params: {}'.format(utils.compute_n_params(model)))

    # testing
    aves_keys = ['vl', 'va']
    aves = {k: utils.Averager() for k in aves_keys}

    test_epochs = args.test_epochs
    np.random.seed(0)
    va_lst = []
    print("epoch|acc|conf_int|loss")
    for epoch in range(1, test_epochs + 1):
        for data, _ in tqdm(loader, leave=False):
            x_shot, x_query = fs.split_shot_query(
                    data.cuda(), n_way, n_shot, n_query,
                    ep_per_batch=ep_per_batch)

            if args.epsilon != 0:
                # perform epsilon corruption
                true_n_shot = x_shot.size()[0] # correct an off-by-one
                assert true_n_shot == 5
                ep_frac = math.floor(args.epsilon * true_n_shot)
                assert ep_frac != 0
                corrupt_idxs = random.sample(range(n_shot), math.floor(args.epsilon * n_shot))
                for idx in corrupt_idxs:
                    # generate a random spherical gaussian noise tensor to add to shot tensor
                    corruption_tensor = torch.normal(mean=0, 
                        std=torch.full(size=x_shot[idx].size(), fill_value=args.std, device=torch.device('cuda:0')))
                    x_shot[idx] = x_shot[idx] + corruption_tensor


            with torch.no_grad():
                if not args.sauc:
                    logits = model(x_shot, x_query).view(-1, n_way)
                    label = fs.make_nk_label(n_way, n_query,
                            ep_per_batch=ep_per_batch).cuda()
                    loss = F.cross_entropy(logits, label)
                    acc = utils.compute_acc(logits, label)

                    aves['vl'].add(loss.item(), len(data))
                    aves['va'].add(acc, len(data))
                    va_lst.append(acc)
                else:
                    x_shot = x_shot[:, 0, :, :, :, :].contiguous()
                    shot_shape = x_shot.shape[:-3]
                    img_shape = x_shot.shape[-3:]
                    bs = shot_shape[0]
                    p = model.encoder(x_shot.view(-1, *img_shape)).reshape(
                            *shot_shape, -1).mean(dim=1, keepdim=True)
                    q = model.encoder(x_query.view(-1, *img_shape)).view(
                            bs, -1, p.shape[-1])
                    p = F.normalize(p, dim=-1)
                    q = F.normalize(q, dim=-1)
                    s = torch.bmm(q, p.transpose(2, 1)).view(bs, -1).cpu()
                    for i in range(bs):
                        k = s.shape[1] // 2
                        y_true = [1] * k + [0] * k
                        acc = roc_auc_score(y_true, s[i])
                        aves['va'].add(acc, len(data))
                        va_lst.append(acc)

        print('{}|{:.4f}|{:.4f}|{:.4f}'.format(
                epoch, aves['va'].item(),
                mean_confidence_interval(va_lst),
                aves['vl'].item()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/test_few_shot.yaml')
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--test-epochs', type=int, default=10)
    parser.add_argument('--sauc', action='store_true')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--epsilon', type=float, default=0.0)
    parser.add_argument('--std', type=float, default=20.0)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True

    utils.set_gpu(args.gpu)
    main(config, args)


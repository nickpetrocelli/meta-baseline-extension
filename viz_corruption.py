# Inefficient script to visualize some things for the report

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
from matplotlib import pyplot as plt


def main(config, args):
    # dataset
    dataset = datasets.make(config['dataset'], **config['dataset_args'])
    utils.log('dataset: {} (x{}), {}'.format(
            dataset[0][0].shape, len(dataset), dataset.n_classes))
    n_way = 5
    n_shot, n_query = args.shot, 1
    n_batch = 200
    ep_per_batch = 4
    batch_sampler = CategoriesSampler(
            dataset.label, n_batch, n_way, n_shot + n_query,
            ep_per_batch=ep_per_batch)
    loader = DataLoader(dataset, batch_sampler=batch_sampler,
                        num_workers=8, pin_memory=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model
    encoder = models.load(torch.load('./save/classifier_mini-imagenet_resnet12/epoch-last.pth')).encoder

    
    for data, _ in tqdm(loader, leave=False):
        x_shot, x_query = fs.split_shot_query(
                data.cuda(), n_way, n_shot, n_query,
                ep_per_batch=ep_per_batch)
            
        ep_frac = math.floor(args.epsilon * n_shot)
        assert ep_frac != 0

        orig_size = x_shot.size()

        # grab a couple images and visualize corruption

        for episode in range(1):
            for way in range(1):
                corrupt_idxs = random.sample(range(n_shot), ep_frac)
                for idx in corrupt_idxs:
                    # generate a random spherical gaussian noise tensor to add to shot tensor
                    corruption_tensor = torch.normal(mean=0, 
                        std=torch.full(size=x_shot[episode][way][idx].size(), fill_value=args.std, device=device))
                    pre_corrupt = x_shot[episode][way][idx]
                    # add dummy batch dim
                    pre_enc = encoder(pre_corrupt[None, :])
                   
                    post_corrupt = x_shot[episode][way][idx] + corruption_tensor
                    post_enc = encoder(post_corrupt[None, :])

                    print("cosine difference: ", {torch.bmm(F.normalize(pre_corrupt, dim=-1),
                               F.normalize(post_corrupt, dim=-1).permute(0, 2, 1))})
                    print("euclidean difference: ", {-(pre_corrupt.unsqueeze(2) -
                       post_corrupt.unsqueeze(1)).pow(2).sum(dim=-1)})
                    #.permute(1, 2, 0)
                    plt.imsave(f'imgs/{idx}_pre_corrupt.png',pre_corrupt.type(dtype=torch.uint8).cpu().numpy()  )
                    plt.imsave(f'imgs/{idx}_post_corrupt.png',post_corrupt.type(dtype=torch.uint8).cpu().numpy()  )
                    






        assert orig_size == x_shot.size(), f'original size: {orig_size}, new size: {x_shot.size()}'
        break




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/test/test_few_shot_mini_20shot_cos.yaml')
    parser.add_argument('--shot', type=int, default=20)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--std', type=float, default=20.0)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True

    utils.set_gpu(args.gpu)
    main(config, args)


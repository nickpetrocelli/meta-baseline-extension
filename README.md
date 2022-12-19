# Few-Shot Meta Baseline with Robust Mean Estimation

Extension of the code for [Meta-Baseline: Exploring Simple Meta-Learning for Few-Shot Learning](https://arxiv.org/abs/2003.04390). Original citation for that paper:
@inproceedings{chen2021meta,
  title={Meta-Baseline: Exploring Simple Meta-Learning for Few-Shot Learning},
  author={Chen, Yinbo and Liu, Zhuang and Xu, Huijuan and Darrell, Trevor and Wang, Xiaolong},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={9062--9071},
  year={2021}
}

## Setup Details
See README_orig.md for Chen et al.'s original instructions on how to set up datasets and run training and testing at a basic level.

The Configs/ folder contains all of the configuration files we used to train Meta-Baseline and evaluate it.

## Primary results
This repo implements random corruption of an epsilon-fraction of the support set at testing time. To access this feature, run `test_few_shot.py` with the option `--epsilon <epsilon fraction>`. Optionally, use the option `--std <standard deviation>` to modify the standard definition of the Gaussian random noise added to input examples' pixels; the default is 20.

We find an epsilon fraction of 0.2 heavily reduces model acccuracy on both MiniImageNet and TieredImageNet. 

Additionally, we augment Meta-Baseline with [High-Dimensional Robust Mean Estimation via Gradient Descent
](https://arxiv.org/abs/2005.01378)

Citation for that paper:
@inproceedings{cheng2020high,
  title={High-dimensional robust mean estimation via gradient descent},
  author={Cheng, Yu and Diakonikolas, Ilias and Ge, Rong and Soltanolkotabi, Mahdi},
  booktitle={International Conference on Machine Learning},
  pages={1768--1778},
  year={2020},
  organization={PMLR}
}

To utilize this when evaluating, run `test_few_shot.py` with the option `--use-pgd`.

The code used for the robust mean estimation algorithm is a port of the code found in https://github.com/chycharlie/robust-bn-faster. Credit for the original implementation goes to Yu Cheng for that repository.

All of our test results can be found in `test_results`.
# NESS

Github Repository for paper **"Learning in the Null Space: Small Singular Values for Continual Learning"** (CPAL 2026 Oral) 

[[Paper]](https://arxiv.org/pdf/2602.21919)

## Abstract

Alleviating catastrophic forgetting while enabling further learning is a primary challenge in continual learning (CL). Orthogonal-based training methods have gained attention for their efficiency and strong theoretical properties, and many existing approaches enforce orthogonality through gradient projection. In this paper, we revisit orthogonality and exploit the fact that small singular values correspond to directions that are nearly orthogonal to the input space of previous tasks. Building on this principle, we introduce NESS (Null-space Estimated from Small Singular values), a CL method that applies orthogonality directly in the weight space rather than through gradient manipulation. Specifically, NESS constructs an approximate null space using the smallest singular values of each layer’s input representation and parameterizes task-specific updates via a compact low-rank adaptation (LoRA-style) formulation constrained to this subspace. The subspace basis is fixed to preserve the null-space constraint, and only a single trainable matrix is learned for each task. This design ensures that the resulting updates remain approximately in the null space of previous inputs while enabling adaptation to new tasks. Our theoretical analysis and experiments on three benchmark datasets demonstrate competitive performance, low forgetting, and stable accuracy across tasks, highlighting the role of small singular values in continual learning.

## Datasets
For experiments with "CIFAR-100" and "5-datasets", datasets will be automatically downloaded to the `data` folder. 

For the MiniImageNet dataset, please follow the guidance of [LYang-666/TRGP](https://github.com/LYang-666/TRGP) and download the `train.pkl` and `test.pkl` to the `data` folder before running the script.

## Packages
The code for NESS (and the baselines) only uses **torch** and **torchvision** .

## NESS usages

Example for experimental scripts can be found in `scripts.sh` script. Feel free to change the hyperparameters. If you want to automatically save logs to `logs` folder, please uncomment these 3 lines in `create_log_dir` function of each python file:

```
# fh = logging.FileHandler(path+'/'+filename) --> UNCOMMENT THIS
# fh.setLevel(logging.DEBUG) --> UNCOMMENT THIS
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# logger.addHandler(fh) --> UNCOMMENT THIS
```
and also make sure your `main` function should look like this if you want autosaved logs:
```
if __name__ == "__main__":
    ...
    str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    log = create_log_dir(args.savename, f'log_{str_time_}.txt')
```

## Baseline usages
For baseline codes (in `baselines` folder), please move the script file needed to the root folder (same folder as other `main_ness_{}.py` code) and then run with the example script in `scripts.sh`. 

For TRGP and FS-DGPM baselines, please clone their repositories and run follow their guidance, especially for TRGP: after cloning the repository, please copy our code in `baselines/TRGP` folder to TRGP repository to reconstruct the results.

## Acknowledgements
Our implementations and baselines acknowledgements and references to these repositories: 

[EnnengYang/DFGP](https://github.com/EnnengYang/DFGP), [sahagobinda/SGP](https://github.com/sahagobinda/SGP), [sahagobinda/GPM](https://github.com/sahagobinda/GPM), [LYang-666/TRGP](https://github.com/LYang-666/TRGP) and [danruod/FS-DGPM](https://github.com/danruod/FS-DGPM)

## Citation
```
@inproceedings{
pham2026learning,
title={Learning in the Null Space: Small Singular Values for Continual Learning},
author={Cuong Anh Pham and Praneeth Vepakomma and Samuel Horv{\'a}th},
booktitle={The Third Conference on Parsimony and Learning (Proceedings Track)},
year={2026},
url={https://openreview.net/forum?id=ZFNMp3Zuo7}
}
```

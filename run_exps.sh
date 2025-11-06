#!/bin/bash
python main_our_cifar100_scratch.py --n_epochs 200 2>&1 | tee -a "logs/cf100_ours_$(date +%F)_scratch.log"
python main_our_cifar100_v1_bug.py --n_epochs 200 2>&1 --eps_1 0.001 | tee -a "logs/cf100_our_$(date +%F)_parallel_v1.log"
python main_our_cifar100_v2.py --n_epochs 200 2>&1 --eps_1 0.001 | tee -a "logs/cf100_our_$(date +%F)_parallel_v2.log"
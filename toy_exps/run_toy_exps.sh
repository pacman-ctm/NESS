#!/bin/bash

# python toy_splitMNIST_original.py 2>&1 | tee -a "logs/toy_exps_$(date +%F).log"

python toy_splitMNIST_svd.py 2>&1 | tee -a "logs/toy_exps_$(date +%F).log"

# python toy_splitMNIST_svd.py
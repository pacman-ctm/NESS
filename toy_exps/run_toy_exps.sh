#!/bin/bash

echo "Running: python toy_splitMNIST_original.py" | tee -a "logs/toy_exps_$(date +%F).log"
python toy_splitMNIST_original.py 2>&1 | tee -a "logs/toy_exps_$(date +%F).log"

echo "Running: python toy_splitMNIST_svd.py" | tee -a "logs/toy_exps_$(date +%F).log"
python toy_splitMNIST_svd.py 2>&1 | tee -a "logs/toy_exps_$(date +%F).log"

# python toy_splitMNIST_svd.py
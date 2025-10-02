#!/bin/bash

# python toy_splitMNIST_original.py 

python toy_splitMNIST_svd.py 2>&1 | tee -a toy_exps_logs.log
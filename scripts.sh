# NESS (both using SAM or SGDm), feel free to modify hyperparameters as needed!

python main_ness_cifar100_sam.py --n_epochs 200 --eps_1 0.001 --lr 0.05 2>&1 | tee -a "logs/ness_cf100_$(date +%F)_sam.log"
python main_ness_cifar100_sgd.py --optimizer sgdm --n_epochs 200 --eps_1 0.001 --lr 0.005 2>&1 | tee -a "logs/ness_cf100_$(date +%F)_sgdm.log" 

python main_ness_fivedataset_sam.py --n_epochs 200 --eps_1 0.001 --lr 0.1 --weight_decay 0.00005 2>&1 | tee -a "logs/ness_5d_$(date +%F)_sam.log"
python main_ness_fivedataset_sgd.py --optimizer sgdm --n_epochs 200 --eps_1 0.001 --lr 0.05 --weight_decay 0.00005 2>&1 | tee -a "logs/ness_5d_$(date +%F)_sgdm.log"

python main_ness_miniimagenet_sam.py --n_epochs 100 --eps_1 0.0005 --lr 0.1 --weight_decay 0.00001 2>&1 | tee -a "logs/ness_mini_$(date +%F)_sam.log"
python main_ness_miniimagenet_sgd.py --optimizer sgdm --n_epochs 100 --eps_1 0.0005 --lr 0.01 --weight_decay 0.00001 2>&1 | tee -a "logs/ness_mini_$(date +%F)_sgdm.log"


# DFGP baselines:

# python main_dfgp_cifar100.py 2>&1 | tee -a "logs/BL_dfpg_cf100_$(date +%F)_ori.log"
# python main_dfgp_fivedataset.py 2>&1 | tee -a "logs/BL_dfpg_5d_$(date +%F)_ori.log"
# python main_dfgp_miniimagenet.py 2>&1 | tee -a "logs/BL_dfpg_mini_$(date +%F)_ori.log"

# SGP baselines:
# python -u main_sgp_cifar100.py 2>&1 | tee -a "logs/BL_sgp_cf100_$(date +%F).log"
# python -u main_sgp_fivedataset.py 2>&1 | tee -a "logs/BL_sgp_5d_$(date +%F).log"
# python -u main_sgp_miniimagenet.py 2>&1 | tee -a "logs/BL_sgp_mini_$(date +%F).log"

# GPM baselines:
# python -u main_gpm_cifar100.py 2>&1 | tee -a "logs/BL_gpm_cf100_$(date +%F).log"
# python -u main_gpm_fivedataset.py 2>&1 | tee -a "logs/BL_gpm_5d_$(date +%F).log"
# python -u main_gpm_miniimagenet.py 2>&1 | tee -a "logs/BL_gpm_mini_$(date +%F).log"


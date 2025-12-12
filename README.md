# NESS

Github Repository for submission paper **"Learning in the Null Space: Small Singular Values for Continual Learning"**.

## Datasets


The dataset for CIFAR-100, 5-dataset will be automatically downloaded to the `data` folder. 

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
For baseline codes (in `baselines` folder), please move them to the root folder (same folder as other `main_ness_{}.py` code) and then run with the example script in `scripts.sh`. 

For TRGP and FS-DGPM baselines, please clone their repositories and run follow their guidance, especially for TRGP: after clone the repository, please copy our code in `baselines/TRGP` folder to TRGP repository to reconstruct the results.

## Acknowledgement
Our implementations and baselines acknowledgements and references to these repositories: 

[EnnengYang/DFGP](https://github.com/EnnengYang/DFGP), [sahagobinda/SGP](https://github.com/sahagobinda/SGP), [sahagobinda/GPM](https://github.com/sahagobinda/GPM), [LYang-666/TRGP](https://github.com/LYang-666/TRGP) and [danruod/FS-DGPM](https://github.com/danruod/FS-DGPM)

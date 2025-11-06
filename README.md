# OrthoCL

## Usage

For environment:
* The code only use **torch** and **torchvision**.

For datasets:
* With CIFAR-100, the dataset is automatically installed during the training process.

For experiments with CIFAR-100:
* Train the continual learning from scratch (to see the maximum accuracy on every tasks and do not care about the forgetting):
```
python main_our_cifar_100_scratch.py
```

* Train the continual learning (change the train function from DFGP), full GPU usage (but is bugged because of using trainable parameter for every layer included batch_norm and biases):
```
python main_our_cifar_100_v1_bug.py
```

* Train the continual learning by with the version adapted from DFGP (slow because still use CPU in most parts and wrong in weight W after every task):
```
python main_our_cifar_100_v2.py
```

* The default code has the logs for training process in "./logs/xxx/log_date.txt" with xxx = {DATASET}.


## Acknowledgement
This code is based on [EnnengYang/DFGP](https://github.com/EnnengYang/DFGP).
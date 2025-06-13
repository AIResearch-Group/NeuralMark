# Hashed Watermark as a Filter: Defending Forging and Overwriting Attacks of Weight-based Neural-Network Watermarking
This is the official implementation codes for Hashed Watermark as a Filter: Defending Forging and Overwriting Attacks of Weight-based Neural-Network Watermarking.

## Getting Started
### Datasets
Datasets will be download automatically. Experiments are done on 5 various datasets, including CIFAR-10, CIFAR-100, Caltech-101, Caltech-256, TinyImageNet.

The dataset folder structure should look like:
```
.
|-- caltech-101
|   `-- 101_ObjectCategories
|-- caltech-256
|   `-- 256_ObjectCategories
|-- cifar10
|   `-- cifar-10-batches-py
|-- cifar100
|   `-- cifar-100-python
`-- imagenet
    `-- TinyImageNet
```

### Dependencies
You can implement the experiment environment by using the instruction below :
```bash
pip install -r env.txt
```

## Evaluations
### Training
Taking ResNet-18 as an example, you can run the following to train watermarked models.
```commandline
bash ./scripts/train.sh
```
This script will automatically train Resnet-18 with NeuralMark on CIFAR-10, CIFAR-100, Caltech-101, Caltech256 and TinyImageNet. Training details will be shown in `logs/`

### Finetune Attack
Run the following to reproduce finetune attack.
```commandline
bash ./scripts/finetune.sh
```
This script will automatically finetune the watermarked models between CIFAR-10, CIFAR-100, Caltech-101 and Caltech-256 for ResNet-18.

### Pruning Attack
Run the following to reproduce pruning attack.
```commandline
bash ./scripts/prune.sh
```
This script will automatically prune the watermarked models with pruning rate as 0.2. If you want to customize the pruning strength, please alter the `--pruning-rate` argument in `scripts/prune.sh`.

### Forging Attacking
Run the following to reproduce forging attack.
```commandline
bash ./scripts/forge.sh
```
This script will automatically forge the finetuned and pruned models on CIFAR-10 and CIFAR-100 with attacker's seed as 110. You can alter the `--watermark-seed` argument in `scripts/forge.sh` to use other pairs of attacking key and watermark. Also, you can change the `--pretrained-path` argument to decide forging on which model.

### Overwriting Attack
Run the following to reproduce overwriting attack.
```commandline
bash ./scripts/overwrite.sh
```
This script will automatically overwrite the watermarked models between CIFAR-10, CIFAR-100, Caltech-101 and Caltech-256 with learning rate as 0.001 and lambda as 1. If you want to customize the overwriting strength, please alter the `--lr` argument and the `--lambda` argument in `scripts/overwrite.sh` respectively.

### NeuralMark for GPT-2 (LoRA)
Switch to `LoRA/` to reproduce the experiment results on GPT-2.


## Copyright Notice

The code in this repository is copyrighted by Beijing Teleinfo Technology Company Ltd., China Academy of Information and Communications Technology.

The software has been registered with the China Copyright Protection Center (Registration Number: 2025SR0859427).

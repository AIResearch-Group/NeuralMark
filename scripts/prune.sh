# Pruning Attack
python classifier.py --action pruning --seed 42 --epochs 1 --pruning-rate 0.2 --pretrained-path logs/resnet_cifar10_train/1/models/final.pth
python classifier.py --action pruning --seed 42 --epochs 1 --pruning-rate 0.2 --pretrained-path logs/resnet_cifar100_train/1/models/final.pth --dataset cifar100
python classifier.py --action pruning --seed 42 --epochs 1 --pruning-rate 0.2 --pretrained-path logs/resnet_caltech101_train/1/models/final.pth --dataset caltech101
python classifier.py --action pruning --seed 42 --epochs 1 --pruning-rate 0.2 --pretrained-path logs/resnet_caltech256_train/1/models/final.pth --dataset caltech256
# Overwrite Attack
python classifier.py --seed 42 --epochs 100 --action overwrite --dataset cifar100 --pretrained-path logs/resnet_cifar10_train/1/models/final.pth --lr 0.001 --watermark-seed 5 --lambda 1
python classifier.py --seed 42 --epochs 100 --action overwrite --pretrained-path logs/resnet_cifar100_train/1/models/final.pth --lr 0.001 --watermark-seed 5 --lambda 1
python classifier.py --seed 42 --epochs 100 --action overwrite --dataset caltech256 --watermark-std 1 --pretrained-path logs/resnet_caltech101_train/1/models/final.pth --lr 0.001 --watermark-seed 5 --lambda 1
python classifier.py --seed 42 --epochs 100 --action overwrite --dataset caltech101 --watermark-std 1 --pretrained-path logs/resnet_caltech256_train/1/models/final.pth --lr 0.001 --watermark-seed 5 --lambda 1
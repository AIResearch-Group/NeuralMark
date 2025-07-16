# Finetune Attack
python classifier.py --seed 42 --epochs 100 --action finetune --dataset cifar100 --pretrained-path logs/resnet_cifar10_train/1/models/final.pth --lr 0.001
python classifier.py --seed 42 --epochs 100 --action finetune --pretrained-path logs/resnet_cifar100_train/1/models/final.pth --lr 0.001
python classifier.py --seed 42 --epochs 100 --action finetune --dataset caltech256 --pretrained-path logs/resnet_caltech101_train/1/models/final.pth --lr 0.001
python classifier.py --seed 42 --epochs 100 --action finetune --dataset caltech101 --pretrained-path logs/resnet_caltech256_train/1/models/final.pth --lr 0.001

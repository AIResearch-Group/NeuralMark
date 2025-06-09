# Forging Attack
python classifier.py --action forging --epochs 1 --watermark-seed 110 --pretrained-path logs/resnet_cifar10_finetune/1/models/final.pth
python classifier.py --action forging --epochs 1 --watermark-seed 110 --pretrained-path logs/resnet_cifar10_pruning/1/models/final.pth
python classifier.py --action forging --epochs 1 --watermark-seed 110 --pretrained-path logs/resnet_cifar100_finetune/1/models/final.pth --dataset cifar100
python classifier.py --action forging --epochs 1 --watermark-seed 110 --pretrained-path logs/resnet_cifar100_pruning/1/models/final.pth --dataset cifar100
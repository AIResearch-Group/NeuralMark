import argparse
from pprint import pprint
from tools.classification import Classification

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--action', default='train', choices=['train', 'finetune', 'pruning', 'forging', 'overwrite', 'baseline'],
                        help='experiment type (default: train)')

    # training
    parser.add_argument('--seed', type=int, default=42,
                        help='training seed (default: 42)')
    parser.add_argument('--arch', default='resnet', choices=['alexnet', 'resnet', 'VIT', 'vgg', 'swin_v2', 'wide_resnet', 'GoogLeNet', 'mobilenet'],
                        help='model architecture (default: resnet)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size (default: 64)')
    parser.add_argument('--epochs', type=int, required=True,
                        help='experiment epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--scheduler', default='scheduler.json',
                        help='scheduler config json file')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10',
                                                                 'cifar100',
                                                                 'caltech101',
                                                                 'caltech256',
                                                                 'imagenet'],
                        help='experiment dataset (default: cifar10)')

    # watermark
    parser.add_argument('--filter-strength', type=int, default=4,
                        help='number of filtering operations (default: 4)')
    parser.add_argument('--lambda', type=float, default=1,
                        help='coe of watermark reg in loss function (default: 1)')
    parser.add_argument('--watermark-std', type=float, default=1,
                        help='std of watermark matrix (default: 1)')
    parser.add_argument('--watermark-k0', type=int, default=512,
                        help='size of the first dimension of watermark matrix (default: 512)')
    parser.add_argument('--watermark-siglen', type=int, default=256,
                        help='size of sig (default: 256)')
    parser.add_argument('--watermark-seed', type=int, default=10,
                        help='seed for wm generator (default: 10)')
    parser.add_argument('--pruning-rate', type=float, default=0.4,
                        help='prune ratio (default: 0.4)')

    # paths
    parser.add_argument('--pretrained-path',
                        help='path of pretrained model')
    

    args = parser.parse_args()

    pprint(vars(args))

    classification = Classification(vars(args))

    classification.training()

    print('The logs can be found at', classification.logdir)
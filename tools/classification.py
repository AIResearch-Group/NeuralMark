import os
from pprint import pprint

import torch
import torchvision
import torch.optim as optim
from torchvision.transforms import transforms
from torchvision.datasets.cifar import CIFAR10, CIFAR100
from tools.dataset import Caltech101, Caltech256, ImageNet
from torch.utils.data import DataLoader

from tools.logdb import LogDB
from tools.ks_generator import ks_generator
from tools.embed_loss import EmbedRegularization
from tools.trainer import Trainer
from tools.seed_everything import seed_everything
from models.resnet import ResNet18
from models.alexnet import AlexNet

class Classification(LogDB):
    def __init__(self, args):
        super().__init__(args)
        seed_everything(self.seed)

        self.prepare_dataset()
        self.create_folder()
        self.construct_model()
        self.prepare_reg()
        
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0005)

        if self.scheduler_config["scheduler"] == True:  # if no specify steps, then scheduler = None
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                       self.scheduler_config["steps"], self.scheduler_config["gamma"])
        else:
            scheduler = None

        self.trainer = Trainer(optimizer, scheduler, self.device)

    def construct_model(self):
        print('Loading arch: ' + self.arch)

        if self.arch == 'alexnet':
            model = AlexNet(in_channels=3, num_classes=self.num_classes)
        elif self.arch == 'resnet':
            model = ResNet18(num_classes=self.num_classes)
        elif self.arch == 'VIT':
            model = torchvision.models.vit_b_32(image_size=32, num_classes=self.num_classes)
        elif self.arch == 'vgg':
            model = torchvision.models.vgg16_bn(num_classes=self.num_classes)
        elif self.arch == 'swin_v2':
            model = torchvision.models.swin_v2_b(num_classes=self.num_classes)
        elif self.arch == 'wide_resnet':
            model = torchvision.models.wide_resnet50_2(num_classes=self.num_classes)
        elif self.arch == 'GoogLeNet':
            model = torchvision.models.googlenet(num_classes=self.num_classes)
        elif self.arch == 'mobilenet':
            model = torchvision.models.mobilenet_v3_large(num_classes=self.num_classes)
        else:
            raise Exception('Unknown arch')

        if self.pretrained_path is not None:
            sd = torch.load(self.pretrained_path)
            if self.action in ['finetune', 'overwrite']:
                print('Changing classifier ...')
                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in sd.items() if k.find('classifier') == -1}
                # update
                model_dict.update(pretrained_dict)
                sd = model_dict
            model.load_state_dict(sd)
        self.model = model.to(self.device)
        pprint(self.model)

    def prepare_reg(self):
        if not os.path.exists('./key_workshop/std{}/seed{}/sig_{}_{}.pt'.format(self.wm_config["std"],
                                                                                self.wm_config["seed"],
                                                                                self.wm_config["k0"],
                                                                                self.wm_config["siglen"])):
            ks_generator(self.wm_config["seed"],
                         self.wm_config["std"],
                         self.wm_config["k0"],
                         self.wm_config["siglen"])

        self.key = torch.load('./key_workshop/std{}/seed{}/key_{}_{}.pt'.format(self.wm_config["std"],
                                                                                self.wm_config["seed"],
                                                                                self.wm_config["k0"],
                                                                                self.wm_config["siglen"])).to(self.device)
        self.sig = torch.load('./key_workshop/std{}/seed{}/sig_{}_{}.pt'.format(self.wm_config["std"],
                                                                                self.wm_config["seed"],
                                                                                self.wm_config["k0"],
                                                                                self.wm_config["siglen"])).to(self.device)

        if self.action_config["reg_new"]:
            self.reg = EmbedRegularization(key=self.key,
                                           sig=self.sig,
                                           layer_name=self.wm_arch_config["wm_layer"],
                                           lambda_=self.lambda_,
                                           filter_num=self.filter_num,
                                           device=self.device)
            self.save_model('final.pth.reg', model=self.reg)
        if self.action_config["reg_old"]:
            self.history_reg = EmbedRegularization(key=self.key,
                                                   sig=self.sig,
                                                   layer_name=self.wm_arch_config["wm_layer"],
                                                   lambda_=self.lambda_,
                                                   filter_num=self.filter_num,
                                                   device=self.device)
            if os.path.exists(f'{self.pretrained_path}.reg'):
                self.history_reg.load_state_dict(torch.load(f'{self.pretrained_path}.reg'))

    def training(self):
        best_acc = float('-inf')
        history_file = os.path.join(self.logdir, 'history.csv')
        first = True

        for ep in range(1, self.epochs + 1):
            if self.action in ['train', 'finetune', 'overwrite', 'baseline']:
                train_metrics = self.trainer.train(self.train_loader, self.model, self.reg if self.action in ['train', 'overwrite'] else None)
            elif self.action == 'pruning':
                self.trainer.pruning(self.model, self.wm_arch_config["wm_layer"], self.pruning_rate)

            valid_metrics = self.trainer.test(self.test_loader, self.model, self.reg if self.action_config["reg_new"] else None, self.history_reg if self.action_config["reg_old"] else None)

            metrics = {}
            if self.action in ['train', 'finetune', 'overwrite', 'baseline']:
                for key in train_metrics:
                    metrics[f'train_{key}'] = train_metrics[key]
            for key in valid_metrics:
                metrics[f'valid_{key}'] = valid_metrics[key]

            self.append_history(history_file, metrics, first)
            first = False

            if best_acc < metrics['valid_acc']:
                best_acc = metrics['valid_acc']
                self.save_model('best.pth')
            for key, value in metrics.items():
                print(f'{key}: {value:6.4f}', end=', ')
            print(f'Best Acc: {best_acc:6.4f}, Epoch: {ep}/{self.epochs}', end='\r')
            
        self.save_model('final.pth')
        

    def prepare_dataset(self):
        ds = self.dataset

        is_cifar = 'cifar' in ds
        root = f'dataset/{ds}'
        print('Loading dataset: ' + ds)

        selected_dataset = {
            'cifar10': CIFAR10,
            'cifar100': CIFAR100,
            'caltech101': Caltech101,
            'caltech256': Caltech256,
            'imagenet': ImageNet
        }[ds]

        self.num_classes = {
            'cifar10': 10,
            'cifar100': 100,
            'caltech101': 102,
            'caltech256': 257,
            'imagenet': 200,
        }[ds]

        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

        # train transform
        if not is_cifar:
            transform_list = [
                transforms.Resize(32),
                transforms.CenterCrop(32)
            ]
        else:
            transform_list = []

        transform_list.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_transforms = transforms.Compose(transform_list)

        # test transform
        if not is_cifar:
            transform_list = [
                transforms.Resize(32),
                transforms.CenterCrop(32)
            ]
        else:
            transform_list = []

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        test_transforms = transforms.Compose(transform_list)

        # dataset and loader
        train_dataset = selected_dataset(root,
                                         train=True,
                                         transform=train_transforms,
                                         download=True)
        test_dataset = selected_dataset(root,
                                        train=False,
                                        transform=test_transforms)
        loader_worker = 4
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=loader_worker,
                                  drop_last=True)
        test_loader = DataLoader(test_dataset,
                                 batch_size=self.batch_size * 2,
                                 shuffle=False,
                                 num_workers=loader_worker)
        self.train_loader = train_loader
        self.test_loader = test_loader

import time

import torch
import torch.nn.functional as functional
import numpy as np

class Trainer(object):
    def __init__(self, optimizer, scheduler, device):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train(self, dataloader, model, regularization, fixed=False):
        model.train()
        embed_loss_meter, loss_meter = 0, 0

        start_time = time.time()
        for i, (data, target) in enumerate(dataloader):
            data = data.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()

            pred = model(data)
            loss = torch.tensor(0.).to(self.device) if fixed else functional.cross_entropy(pred, target)
            embed_loss = torch.tensor(0.).to(self.device)

            # add up reg loss
            if regularization is not None:
                embed_loss = regularization(model).to(self.device)

            (loss + embed_loss).backward()
            self.optimizer.step()

            embed_loss_meter += embed_loss.item()
            loss_meter += loss.item()

        if self.scheduler is not None:
            self.scheduler.step()

        return {'loss': loss_meter / len(dataloader),
                'embed_loss': embed_loss_meter / len(dataloader),
                'time': time.time() - start_time}

    def test(self, dataloader, model, regularization, history_regularization):
        model.eval()
        loss_meter, acc_meter, count, embed_acc, history_embed_acc = 0, 0, 0, 0, 0

        start_time = time.time()
        with torch.no_grad():
            for load in dataloader:
                data, target = load[:2]
                data = data.to(self.device)
                target = target.to(self.device)

                pred = model(data)
                loss_meter += functional.cross_entropy(pred, target, reduction='sum').item()
                pred = pred.max(1, keepdim=True)[1]

                acc_meter += pred.eq(target.view_as(pred)).sum().item()
                count += data.size(0)

            if regularization is not None:
                embed_acc = regularization(model, accuracy=True)
            if history_regularization is not None:
                history_embed_acc = history_regularization(model, accuracy=True)

        return {'loss': loss_meter / count,
                'acc': 100 * acc_meter / count,
                'embed_acc': embed_acc,
                'history_embed_acc': history_embed_acc,
                'time': time.time() - start_time}
    
    def pruning(self, model, layer_name, prune_ratio=0.4):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in layer_name:
                    weight = param
            num_prune = int(prune_ratio * weight.numel())
            prune_indices = np.random.choice(weight.numel(), num_prune, replace=False)
            flat_weight = weight.view(-1)
            flat_weight[prune_indices] = 0
            weight.data = flat_weight.view_as(weight)
        
        print("Pruning done!")

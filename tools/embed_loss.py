import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbedRegularization(nn.Module):
    def __init__(self, key, sig, layer_name, lambda_, filter_num, device, threshold=0.5):
        super(EmbedRegularization, self).__init__()
        self.lambda_ = lambda_
        self.filter_num = filter_num
        if torch.is_tensor(key):
            self.register_buffer('key', key)
        self.register_buffer('sig', sig)
        self.threshold = threshold
        self.layer_name = layer_name
        self.device = device
        print(f'Filter {self.filter_num}, Lambda {self.lambda_}')
    
    def extract_param(self, model):
        for name, param in model.named_parameters():
            if name in self.layer_name:
                weight = param
                sig_size = self.sig.numel()
                for i in range(self.filter_num):
                    weight = weight.view(-1)[:(weight.numel() // sig_size) * sig_size]
                    S_repeated = self.sig.repeat((weight.numel() // sig_size))
                    mask = S_repeated == 1
                    weight = torch.masked_select(weight, mask)
                    
                valid_weight = weight[:(weight.numel() // self.key.shape[0]) * self.key.shape[0]]
                avg_weight = valid_weight.view(-1, self.key.shape[0]).mean(dim=0)
                return avg_weight

    def forward(self, model, accuracy=False):
        weight_extraction = self.extract_param(model)
        pred_sig = torch.matmul(weight_extraction, self.key)
        
        if accuracy:
            res = torch.sigmoid(pred_sig)
            res_binary = torch.where(res.to(self.device) > 0.5, torch.tensor(1.0, device=self.device), torch.tensor(0.0, device=self.device))
            correct = torch.sum(res_binary == self.sig.to(self.device))
            return correct.item() / self.sig.numel()
        else:
            return self.lambda_ * F.binary_cross_entropy_with_logits(pred_sig, self.sig)

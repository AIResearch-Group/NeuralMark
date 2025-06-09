import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbedRegularization(nn.Module):
    def __init__(self, key, sig, layer, lambda_, filter_num, device, threshold=0.5):
        super(EmbedRegularization, self).__init__()
        self.lambda_ = lambda_
        self.filter_num = filter_num
        if torch.is_tensor(key):
            self.register_buffer('key', key)
        self.register_buffer('sig', sig)
        self.threshold = threshold
        self.layer = layer
        self.device = device
        #print(f'Filter {self.filter_num}, Lambda {self.lambda_}')
    
    def extract_param(self):
        weight = self.layer
        sig_size = self.sig.numel()
        for i in range(self.filter_num):
            weight = weight.reshape(-1)[:(weight.numel() // sig_size) * sig_size]
            S_repeated = self.sig.repeat((weight.numel() // sig_size))

            #weight = weight.reshape(-1)[S_repeated == 1]
            mask = S_repeated == 1
            weight = torch.masked_select(weight.reshape(-1), mask)
                    
        valid_weight = weight[:(weight.numel() // self.key.shape[0]) * self.key.shape[0]]
        avg_weight = valid_weight.view(-1, self.key.shape[0]).mean(dim=0)

        return avg_weight

    def forward(self, accuracy=False):
        weight_extraction = self.extract_param()
        pred_sig = torch.matmul(weight_extraction, self.key)
        
        if accuracy:
            res = torch.sigmoid(pred_sig)
            res_binary = torch.where(res.to(self.device) > 0.5, torch.tensor(1.0, device=self.device), torch.tensor(0.0, device=self.device))
            correct = torch.sum(res_binary == self.sig.to(self.device))
            #correct = torch.sum(torch.abs(res.to(self.device) - self.sig.to(self.device)) < self.threshold)
            return correct.item() / self.sig.numel()
        else:
            return self.lambda_ * F.binary_cross_entropy_with_logits(pred_sig, self.sig)

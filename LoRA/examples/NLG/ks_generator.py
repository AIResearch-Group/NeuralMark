import torch
import os
import math
import hashlib
from src.seed_everything import seed_everything

class ks_generator():
    def __init__(self, seed, std, k0, siglen):
        self.seed = seed
        seed_everything(seed)
        self.std = std
        self.k0 = k0
        self.siglen = siglen

        self.key_generator()
        self.sign_generator()

    def key_generator(self):
        self.key = torch.normal(size=(self.k0, self.siglen), mean = 0.0, std = self.std)
        #matrix = torch.rand(size = (data_n, n))
        os.makedirs('./key_workshop/std{}/seed{}'.format(self.std, self.seed), exist_ok=True)
        torch.save(self.key, './key_workshop/std{}/seed{}/key_{}_{}.pt'.format(self.std, self.seed, self.k0, self.siglen))
        
    def sign_generator(self):
        hasher = hashlib.shake_256()
        hasher.update(self.key.cpu().numpy().tobytes())
        hash_bytes = hasher.digest(int(self.siglen / 8))

        hash_val = ''.join(format(byte, '08b') for byte in hash_bytes)
        tensor_val = []
        for i in range(self.siglen):
            if(hash_val[i] == '0'):
                tensor_val.append(0.0)
            else:
                tensor_val.append(1.0)
            
        self.sign = torch.tensor(tensor_val)
        torch.save(self.sign, './key_workshop/std{}/seed{}/sig_{}_{}.pt'.format(self.std, self.seed, self.k0, self.siglen))


if __name__ == '__main__':
    seed = 42
    std = 1
    k0 = 512
    siglen = 256
    ks_generator(seed, std, k0, siglen)


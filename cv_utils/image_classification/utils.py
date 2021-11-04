import numpy as np
import torch
import yaml

class Configs:
    def __init__(self, path_config_file):
        self.configs = yaml.safe_load(open(path_config_file).read())
    
    def __getattr__(self, item):
        return self.configs.get(item, None)

class ToTensor():
    def __init__(self, dtype='float32'):
        self.dtype = dtype
    
    def __call__(self, x, **kwargs):
        return torch.from_numpy(x.transpose(2, 0, 1).astype(self.dtype))
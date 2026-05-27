import torch
from torch.utils.data import DataLoader, Dataset

class dataSetLoaderCus:
    def __init__(self, num_samples, input_dim):
        self.data=torch.randn(num_samples, input_dim)
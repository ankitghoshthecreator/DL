
import torch
from torch.utils.data import DataLoader, Dataset

class dataSetLoaderCus(Dataset):
    def __init__(self, num_samples, input_dim):
        self.data=torch.randn(num_samples, input_dim)
        self.label=torch.randn(num_samples,1)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        sample=self.data[item]
        label=self.label[item]

        return sample, label

dataset=dataSetLoaderCus(100, 100)

dataLoader=DataLoader(
    dataset=dataset,
    batch_size=32,
    shuffle=True
)

for epoch in range(1):
    for batch_idx,(batch_data, batch_label) in enumerate(dataLoader):
        if batch_idx == 0:
            print(f"Batch Data Shape: {batch_data.shape}")
            print(f"Batch Labels Shape: {batch_label.shape}")
            break

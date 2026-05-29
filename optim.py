import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import MSELoss
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

from customLoss import criterion

# 1. Setup Data
X = torch.randn(100, 1)
Y = 3 * X + 0.5 + torch.randn(100, 1) * 0.1 # Synthetic linear data
dataset = TensorDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

class SimpleReg(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear=nn.Linear(1,1)
    def forward(self,x):
        return self.linear(x)
model=SimpleReg()

criterion=nn.MSELoss()
optim=optim.Adam(model.parameters(), lr=.1)

loss_h=[]

for epoch in range(1000):
    epoch_loss=0
    for x_batch, y_batch in dataloader:
        optim.zero_grad()
        output=model(x_batch)
        loss=criterion(output, y_batch)
        loss.backward()
        optim.step()
        epoch_loss+=loss.item()
    loss_h.append(epoch_loss/(len(dataloader)))
# Visualizing the Learning Process
plt.plot(loss_h)
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Training Loss Convergence')
plt.savefig('training_loss.png')
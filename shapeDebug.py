import torch
import torch.nn as nn
from torchgen import model


class shapeDebug(nn.Module):
    def __init__(self):
        super(shapeDebug, self).__init__()
        self.fc1=nn.Linear(50,100)
        self.fc2=nn.Linear(100,10)

    def forward(self,x):
        print("Input shape",x.shape)

        x=self.fc1(x)
        print("Output shape fc1",x.shape)

        x=self.fc2(x)
        print("Output shape fc2",x.shape)
        return x

model=shapeDebug()

try:
    dummy_ip=torch.randn(50,150)
    op=model(dummy_ip)
except RuntimeError as e:
    print(e)

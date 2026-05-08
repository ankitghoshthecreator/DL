import torch
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self, input_size,hidden_size, output_size):
        #defining the layers
        super(CustomModel, self).__init__()

        self.layer1=nn.Linear(input_size, hidden_size)
        self.relu=nn.ReLU()
        self.layer2=nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x=self.layer1(x)
        x=self.relu(x)
        x=self.layer2(x)

        return x
# Instantiate and Test
model = CustomModel(input_size=10, hidden_size=5, output_size=1)
sample_input = torch.randn(1, 10)
output = model(sample_input)
print(output)
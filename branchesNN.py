
import torch
from torch.ao.nn.quantized.functional import leaky_relu
import torch
import torch.nn as nn

class branchesNN(nn.Module):
    def __init__(self, input, hidden):
        super(branchesNN, self).__init__()

        self.pathA=nn.Linear(input, hidden)

        self.pathBLin=nn.Linear(input, hidden)
        self.pathBAct=nn.Sigmoid()

        #after merge
        self.classifier=nn.Linear(hidden*2, 1)

    def forward(self, x):
        op_a=self.pathA(x)
        print(op_a)

        op_b=self.pathBLin(x)
        op_b=self.pathBAct(op_b)
        print(op_b)

        merged=torch.cat((op_a, op_b), 1)
        print(merged)
        f_op=self.classifier(merged)
        return f_op

model = branchesNN(input=16, hidden=8)
dummy_data = torch.randn(4, 16) # Batch size of 4
output = model(dummy_data)
print(output)
print(f"Output shape: {output.shape}") # Should be [4, 1]


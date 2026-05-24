import torch
import torch.nn as nn

from customNN import model


class customLoss(nn.Module):
    def __init__(self, penalty_lossW=0.0):
        super(customLoss, self).__init__()
        self.penalty_lossW=penalty_lossW

    def forward(self, y_pred, y_true):

        loss=torch.abs(y_true-y_pred)
        base_l=loss.mean()

        penalty=self.penalty_lossW*torch.mean(torch.abs(base_l))

        total_l=penalty+base_l

        return total_l
criterion=customLoss(penalty_lossW=0.1)

predictions = torch.tensor([[2.5], [3.8]], requires_grad=True)
targets = torch.tensor([[3.0], [4.0]], requires_grad=False)

# Compute Loss
loss = criterion(predictions, targets)
print(f"Computed Loss Value: {loss.item()}")


import torch
import torch.nn as nn



class LSRm(nn.Module):
    def __init__(self, hidden_size, input_size, output_size, num_layers):
        super(LSRm, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.lstm=nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.fc=nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out,(hn, cn)= self.lstm(x)

        l_t_s=out[:, -1, :]

        f_o=self.fc(l_t_s)
        return f_o

model = LSRm(input_size=20, hidden_size=64, num_layers=2, output_size=5)
dummy_sequence = torch.randn(8, 10, 20)
output = model(dummy_sequence)
print(f"Final output shape: {output.shape}") # Should be [8, 5]












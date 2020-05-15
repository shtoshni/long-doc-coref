import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 num_layers=1, bias=False, drop_module=None, activation='tanh'):
        super(MLP, self).__init__()
        if activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        self.drop_module = drop_module
        self.fc_layers = nn.ModuleList([])
        self.fc_layers.append(nn.Linear(input_size, hidden_size, bias=bias))
        for i in range(num_layers - 1):
            self.fc_layers.append(
                nn.Linear(hidden_size, hidden_size, bias=bias))
        self.fc_layers.append(nn.Linear(hidden_size, output_size, bias=bias))

    def forward(self, x):
        out = x
        for idx, layer in enumerate(self.fc_layers):
            out = layer(out)
            if idx != (len(self.fc_layers) - 1):
                out = self.activation(out)
                if self.drop_module:
                    out = self.drop_module(out)
        return out

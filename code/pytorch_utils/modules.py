import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 num_hidden_layers=1, bias=False, drop_module=None, activation='relu'):
        super(MLP, self).__init__()
        if activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        self.drop_module = drop_module
        self.fc_layers = nn.ModuleList([])
        self.num_hidden_layers = num_hidden_layers

        cur_output_size = input_size
        for i in range(num_hidden_layers):
            self.fc_layers.append(
                nn.Linear(cur_output_size, hidden_size, bias=bias))
            cur_output_size = hidden_size

        self.fc_layers.append(nn.Linear(cur_output_size, output_size, bias=bias))

    def forward(self, mlp_input):
        output = mlp_input
        for idx in range(self.num_hidden_layers):
            cur_layer = self.fc_layers[idx]
            output = self.activation(cur_layer(output))
            if self.drop_module:
                output = self.drop_module(output)

        output = self.fc_layers[-1](output)
        return output

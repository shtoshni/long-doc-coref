import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential


class ModuleWrapperIgnores2ndArg(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self,x, dummy_arg=None):
        assert dummy_arg is not None
        x = self.module(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 num_hidden_layers=1, bias=False, drop_module=None,
                 checkpoint=False):
        super(MLP, self).__init__()
        self.layer_list = []

        self.activation = nn.ReLU()
        self.drop_module = drop_module
        self.num_hidden_layers = num_hidden_layers

        cur_output_size = input_size
        for i in range(num_hidden_layers):
            self.layer_list.append(nn.Linear(cur_output_size, hidden_size, bias=bias))
            self.layer_list.append(self.activation)
            if self.drop_module is not None:
                self.layer_list.append(self.drop_module)
            cur_output_size = hidden_size

        self.layer_list.append(nn.Linear(cur_output_size, output_size, bias=bias))
        self.fc_layers = nn.Sequential(*self.layer_list)

        self.use_checkpoint = checkpoint
        if self.use_checkpoint:
            print("Checkpoint is TRUE !!")
            self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
            self.module_wrapper = ModuleWrapperIgnores2ndArg(self.fc_layers)

    def forward(self, mlp_input, dummy_input=None):
        if self.use_checkpoint:
            output = checkpoint(self.module_wrapper, mlp_input, self.dummy_tensor)
        else:
            output = self.fc_layers(mlp_input)
        return output


# class MLP(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size,
#                  num_hidden_layers=1, bias=False, drop_module=None, activation='relu'):
#         super(MLP, self).__init__()
#         if activation == 'tanh':
#             self.activation = nn.Tanh()
#         else:
#             self.activation = nn.ReLU(inplace=False)
#         self.drop_module = drop_module
#         self.fc_layers = nn.ModuleList([])
#         self.num_hidden_layers = num_hidden_layers
#
#         cur_output_size = input_size
#         for i in range(num_hidden_layers):
#             self.fc_layers.append(
#                 nn.Linear(cur_output_size, hidden_size, bias=bias))
#             cur_output_size = hidden_size
#
#         self.fc_layers.append(nn.Linear(cur_output_size, output_size, bias=bias))
#
#     def forward(self, mlp_input):
#         output = mlp_input
#         for idx in range(self.num_hidden_layers):
#             cur_layer = self.fc_layers[idx]
#             output = self.activation(cur_layer(output))
#             if self.drop_module:
#                 output = self.drop_module(output)
#
#         output = self.fc_layers[-1](output)
#         return output

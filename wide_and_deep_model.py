import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class WideAndDeep(nn.Module):
    
    def __init__(self, input_dim,  wide_dim, hidden_layers, dropout=0, num_classes = 4):
        super(WideAndDeep, self).__init__()
        self.input_dim = input_dim
        self.wide_dim = wide_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.num_classes = num_classes

        # Build the deep model
        self.linear_1 = nn.Linear(self.input_dim, hidden_layers[0])
        if self.dropout:
            self.linear_1_drop = nn.Dropout(dropout)

        for i, h in enumerate(self.hidden_layers[1:], 1):
            setattr(self, 'linear_{}'.format(i+1), nn.Linear(self.hidden_layers[i-1], self.hidden_layers[i]))
            if self.dropout:
                setattr(self, 'linear_{}_drop'.format(i+1), nn.Dropout(dropout))

        # Connect the wide side and the deep side
        self.output_layer = nn.Linear(self.hidden_layers[-1] + wide_dim, self.num_classes)

    def forward(self, x_wide, x_deep):

        # feed forward through the deep network
        deep_output = F.relu(self.linear_1(x_deep))
        if self.dropout:
            deep_output = self.linear_1_drop(deep_output)
        
        for i in range(1, len(self.hidden_layers)):
            deep_output = F.relu(getattr(self, 'linear_{}'.format(i+1))(deep_output))
            if self.dropout:
                deep_output = getattr(self, 'linear_{}_drop'.format(i+1))(deep_output)

        # connect the two sides
        wide_deep_input = torch.cat([deep_output, x_wide.float()], dim=1)

        output = self.output_layer(wide_deep_input)
        output = F.softmax(output, dim=1)

        return output


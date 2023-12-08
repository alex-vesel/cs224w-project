import torch

class LinearEncoder(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearEncoder, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def encode(self, input):
        return self.linear(input)

    def reset_parameters(self):
        self.linear.reset_parameters()
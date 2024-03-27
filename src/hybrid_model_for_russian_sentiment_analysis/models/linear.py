from torch import nn



class LinearBlock(nn.Module):
    
    @property
    def device(self):
        for p in self.parameters():
            return p.device

    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 p: float = 0.):
        super(LinearBlock, self).__init__()

        self.add_module('linear', nn.Linear(input_dim, hidden_dim))
        self.add_module('relu', nn.ReLU())
        self.add_module('dropout', nn.Dropout(p=p))
        self.add_module('norm', nn.BatchNorm1d(hidden_dim))


    def forward(self, x):
        x = self._modules['linear'](x)
        x = self._modules['relu'](x)
        x = self._modules['dropout'](x)
        x = self._modules['norm'](x)
        return x
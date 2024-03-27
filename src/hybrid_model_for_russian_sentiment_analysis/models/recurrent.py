import torch
from torch import nn

from hybrid_model_for_russian_sentiment_analysis.models.linear import LinearBlock



class RecModel(nn.Module):

    RECURRENT_MODELS = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}
    
    @property
    def device(self):
        for p in self.parameters():
            return p.device

    
    def __init__(self, 
                 rec_model_kwargs: dict, 
                 hidden_layers: list = [], 
                 p: float = 0.):
        super(RecModel, self).__init__()

        self.rec_model_name = list(rec_model_kwargs.keys())[0]
        self.bidirectional = rec_model_kwargs[self.rec_model_name]['bidirectional']
        self.add_module(self.rec_model_name, self.RECURRENT_MODELS[self.rec_model_name](**rec_model_kwargs[self.rec_model_name]))

        if len(hidden_layers) > 0:
            self.add_module('ffnn', nn.Sequential())
            
            input_dim = rec_model_kwargs[self.rec_model_name]['hidden_size']
            if self.bidirectional:
                input_dim = input_dim * 2
                
            for it, hidden_dim in enumerate(hidden_layers):
                self._modules['ffnn'].add_module(f'block{it}', LinearBlock(input_dim, hidden_dim, p))
                input_dim = hidden_dim

    
    def forward(self, x):
        x = self._modules[self.rec_model_name](x)[0][:, -1, :]
        if 'ffnn' in self._modules:
            x = self._modules['ffnn'](x)
        return x 
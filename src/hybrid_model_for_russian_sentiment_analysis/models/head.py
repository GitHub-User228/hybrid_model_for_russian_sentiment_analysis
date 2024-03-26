import torch
from torch import nn 
from transformers import AutoModel

from hybrid_model_for_russian_sentiment_analysis.models.recurrent import RecModel
from hybrid_model_for_russian_sentiment_analysis.models.convolutional import ConvModel, ParallelConvModels



class HeadModel(nn.Module):
    
    @property
    def device(self):
        for p in self.parameters():
            return p.device

    
    def __init__(self, 
                 main_model, 
                 output_size: int, 
                 n_targets: int, 
                 main_model_kwargs: dict):
        super(HeadModel, self).__init__()
        
        self.add_module('main_model', main_model(**main_model_kwargs))
        self.add_module('ffnn', nn.Sequential())
        self._modules['ffnn'].add_module('classifier', nn.Linear(output_size, n_targets))
        self._modules['ffnn'].add_module('softmax', nn.Softmax(dim=1))

    
    def forward(self, x):
        x = self._modules['main_model'](x)
        x = self._modules['ffnn'](x)
        return x
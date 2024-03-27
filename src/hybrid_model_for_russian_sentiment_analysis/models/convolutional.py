import math
import torch
from torch import nn
from hybrid_model_for_russian_sentiment_analysis import logger
from hybrid_model_for_russian_sentiment_analysis.models.linear import LinearBlock
        


class Conv1DBlock(nn.Module):
    
    @property
    def device(self):
        for p in self.parameters():
            return p.device

    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 conv_kernel_size: int,
                 conv_padding: int,
                 pool_kernel_size: dict,
                 input_size: dict,
                 dim: str = 'features'):
        super(Conv1DBlock, self).__init__()

        layernorm1d_input_size = input_size[dim] + 2*conv_padding-conv_kernel_size + 1
        self.output_size = {'sequence': out_channels, 'features': out_channels} 
        for (k, v) in pool_kernel_size.items():
            if k == dim:
                self.output_size[k] = math.floor((layernorm1d_input_size - v)/v + 1)
            else:
                self.output_size[k] = math.floor((self.output_size[k] - v)/v + 1)
        self.dim = dim
        
        self.add_module('conv1d', nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=conv_kernel_size, padding=conv_padding))      
        self.add_module('layernorm1d', nn.LayerNorm(normalized_shape=(layernorm1d_input_size)))
        self.add_module('relu', nn.ReLU())
        if dim == 'features':
            self.add_module('maxpool2d', nn.MaxPool2d(kernel_size=[pool_kernel_size['sequence'], pool_kernel_size['features']]))
        elif dim == 'sequence':
            self.add_module('maxpool2d', nn.MaxPool2d(kernel_size=[pool_kernel_size['features'], pool_kernel_size['sequence']]))


    def forward(self, x):
        # print(x.shape, self.dim)
        if self.dim == 'sequence':
            x = torch.permute(x, dims=(0, 2, 1))
            # print(x.shape)
        x = self._modules['conv1d'](x)
        x = self._modules['layernorm1d'](x)
        x = self._modules['relu'](x)
        x = self._modules['maxpool2d'](x)
        if self.dim == 'sequence':
            x = torch.permute(x, dims=(0, 2, 1))
        return x



class ConvModel(nn.Module):
    
    @property
    def device(self):
        for p in self.parameters():
            return p.device

    
    def __init__(self, 
                 blocks_kwargs: dict,
                 sequence_size: int,
                 embedding_size: int,
                 hidden_layers: list = [],
                 p: float = 0.):
        super(ConvModel, self).__init__()

        self.return_feature_map = dict([(k, kw['return_feature_map']) for (k, kw) in blocks_kwargs.items()])

        if not list(self.return_feature_map.values())[-1]:
            logger.error('There is no output required for the last convolutional block. Consider changing return_feature_map to True')
            raise
        
        self.output_size = {'sequence': sequence_size, 'features': embedding_size}
        self.add_module('cnn', nn.ModuleDict())
        
        for it, (k, kw) in enumerate(blocks_kwargs.items()):
            
            self._modules['cnn'].update({k: Conv1DBlock(in_channels=kw['in_channels'], out_channels=kw['out_channels'], 
                                                      conv_kernel_size=kw['conv_kernel_size'], conv_padding=kw['conv_padding'],
                                                      pool_kernel_size=kw['pool_kernel_size'], input_size=self.output_size, dim=kw['dim'])})
            self.output_size = self._modules['cnn'][k].output_size.copy()

        if len(hidden_layers) > 0:

            if (len(blocks_kwargs) > 1) and (sum(self.return_feature_map.values()) > 1):
                logger.error(f'hidden_layers are specified, while more than 1 feature maps are required to be used later.')
                raise

            else:
                self.add_module('ffnn', nn.Sequential())
                self._modules['ffnn'].add_module('flatten', nn.Flatten())
                input_dim = self.output_size['sequence']*self.output_size['features']
                
                for it, hidden_dim in enumerate(hidden_layers):
                    self._modules['ffnn'].add_module(f'block{it}', LinearBlock(input_dim, hidden_dim, p))
                    input_dim = hidden_dim   
                    
                self.output_size = input_dim

    
    def forward(self, x):
        if sum(self.return_feature_map.values()) > 1:
            output = {}
            for block in self._modules['cnn'].keys():
                x = self._modules['cnn'][block](x)
                if self.return_feature_map[block]:
                    output[block] = x
            return output 
        else:
            for block in self._modules['cnn'].keys():
                x = self._modules['cnn'][block](x)
            if 'ffnn' in self._modules:
                x = self._modules['ffnn'](x)  
            return x



class ParallelConvModels(nn.Module):
    
    @property
    def device(self):
        for p in self.parameters():
            return p.device

    
    def __init__(self, 
                 conv_models_kwargs: dict,
                 hidden_layers: list = [],
                 p: float = 0.):
        super(ParallelConvModels, self).__init__()

        self.return_feature_map = {}
        self.output_size = {}
        self.add_module('convolutional_models', nn.ModuleDict())
        for (model_name, model_kwargs) in conv_models_kwargs.items():
            model = ConvModel(**model_kwargs)
            self._modules['convolutional_models'].update({model_name: model})
            self.return_feature_map[model_name] = model.return_feature_map
            self.output_size[model_name] = model.output_size
        

        if len(hidden_layers) > 0:

            if all([len(model_kwargs['blocks_kwargs']) > 1 for model_kwargs in conv_models_kwargs.values()]) and \
               any([sum(v.values()) > 1 for v in self.return_feature_map.values()]):
                logger.error(f'hidden_layers are specified, while more than 1 feature maps are required to be used later for one of the models.')
                raise

            else:
                self.add_module('ffnn', nn.Sequential())
                input_dim = 0
                for k in self._modules['convolutional_models'].keys():
                    input_dim += self._modules['convolutional_models'][k].output_size
                
                for it, hidden_dim in enumerate(hidden_layers):
                    self._modules['ffnn'].add_module(f'block{it}', LinearBlock(input_dim, hidden_dim, p))
                    input_dim = hidden_dim   

                self.output_size = hidden_dim

    
    def forward(self, x):
        output = {}
        for k in self._modules['convolutional_models'].keys():
            output[k] = self._modules['convolutional_models'][k](x)
        if 'ffnn' in self._modules:
            output = torch.cat(list(output.values()), axis=-1)
            output = self._modules['ffnn'](output)  
        return output
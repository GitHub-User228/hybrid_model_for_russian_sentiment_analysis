import torch
from torch import nn
from hybrid_model_for_russian_sentiment_analysis import logger



class SelfAttentionHead(nn.Module):
    
    @property
    def device(self):
        for p in self.parameters():
            return p.device

    
    def __init__(self,
                 n_features: int,
                 query_dim: int = None,
                 key_dim: int = None,
                 normalize: bool = True,
                 residual_connection: bool = True):
        super(SelfAttentionHead, self).__init__()

        self.normalize = normalize
        self.residual_connection = residual_connection
        
        if query_dim == None:
            query_dim = n_features
        if key_dim == None:
            key_dim = n_features

        if query_dim != key_dim:
            logger.error('query_dim is not equal to key_dim')
            raise
        
        self.add_module('query', nn.Linear(n_features, query_dim, bias=False))
        self.add_module('key', nn.Linear(n_features, key_dim, bias=False))
        self.add_module('value', nn.Linear(n_features, n_features, bias=False))
        self.add_module('softmax', nn.Softmax(dim=-1))
        if self.normalize:
            self.add_module('layernorm', nn.LayerNorm((n_features)))


    def forward(self, x):
        
        q = self._modules['query'](x)
        k = self._modules['key'](x)
        v = self._modules['value'](x)
        z = torch.matmul(self._modules['softmax'](torch.matmul(q, k.permute(0, 2, 1))), v)
        if self.residual_connection and self.normalize:
            z = self._modules['layernorm'](x + z)
        if self.residual_connection and not self.normalize:
            z = x + z
        if not self.residual_connection and self.normalize:
            z = self._modules['layernorm'](x) 
        return z



class MultiHeadSelfAttention(nn.Module):
    
    @property
    def device(self):
        for p in self.parameters():
            return p.device

    
    def __init__(self,
                 n_features: int,
                 n_heads: int,
                 query_dims: list = None,
                 key_dims: list = None,
                 p: float = 0.1):
        super(MultiHeadSelfAttention, self).__init__()

        if query_dims == None:
            query_dims = [n_features for _ in range(n_heads)]
        if key_dims == None:
            key_dims = [n_features for _ in range(n_heads)]

        self.n_heads = n_heads
        for it in range(n_heads):
            self.add_module(f'head{it}', SelfAttentionHead(n_features, query_dims[it], key_dims[it], 
                                                           normalize=False, residual_connection=True))

        self.add_module('joiner', nn.Linear(n_features*n_heads, n_features, bias=False))
        self.add_module('dropout', nn.Dropout(p=p))
        self.add_module('layernorm', nn.LayerNorm((n_features)))
        


    def forward(self, x):

        # Calculating output for each head
        z = []
        for it in range(self.n_heads):
            z.append(self._modules[f'head{it}'](x))
            
        # Joining outputs
        z = torch.cat(z, dim=-1)
        z = self._modules['joiner'](z)
        z = self._modules['dropout'](z)

        # Residual connection & normalization
        return self._modules['layernorm'](x + z)
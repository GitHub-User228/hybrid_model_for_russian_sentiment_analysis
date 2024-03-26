import torch



class CustomDataset(torch.utils.data.Dataset):

    
    def __init__(self, input_ids, attention_mask):

        super(CustomDataset, self).__init__()
        
        if isinstance(input_ids, torch.Tensor):
            if input_ids.dtype == torch.int32:
               self.input_ids = input_ids
            else:
               self.input_ids = input_ids.to(torch.int32) 
        else:
            self.input_ids = torch.tensor(input_ids).to(torch.int32)

        if isinstance(attention_mask, torch.Tensor):
            if attention_mask.dtype == torch.int8:
               self.attention_mask = attention_mask
            else:
               self.attention_mask = attention_mask.to(torch.int8) 
        else:
            self.attention_mask = torch.tensor(attention_mask).to(torch.int8)
            

    def __len__(self):
        return self.input_ids.shape[0]

    
    def __getitem__(self, i):
        return self.input_ids[i,:], self.attention_mask[i,:]



class EmbeddingsDataset(torch.utils.data.Dataset):

    
    def __init__(self, X):

        super(EmbeddingsDataset, self).__init__()

        if isinstance(X, torch.Tensor):
            if X.dtype == torch.float32:
               self.X = X
            else:
               self.X = X.to(torch.float32) 
        else:
            self.X = torch.tensor(X).to(torch.float32)
    
    def __len__(self):
        return self.X.shape[0]

    
    def __getitem__(self, i):
        return self.X[i,:]
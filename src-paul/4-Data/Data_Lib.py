import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class Custom_Dataset(Dataset):
    def __init__(self, coords, CLs, HP, validation=False):
        super().__init__()

        coords = torch.tensor(coords).float()

        # Split the coordinates into x and y coordinates Data format is [batch_size, 2, features/2]
        x_i = torch.zeros((coords.shape[0], 2, HP["input_features"]//2)) 
        x_i[:,0,:] = coords[:,:(HP["input_features"]//2)]
        x_i[:,1,:] = coords[:,(HP["input_features"]//2):]
        self.coords = x_i

        self.cls = torch.tensor(CLs).float()
        self.len = len(coords)
        self.validation = validation
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        coord = self.coords[index,:,:]
        cl = self.cls[index]
        
        # Return only the first sample --> Test configuration
        if self.validation:
            coord = self.coords[0,:,:]
            cl = self.cls[0]
        
        return coord, cl

def get_data_loader(dataset, HP):
    
    data_loader = DataLoader(
        dataset, 
        batch_size=HP["batch_size"], 
        shuffle=HP["shuffle"],                # shuffle data before loading
        num_workers=HP["num_workers"],        # number of workers for data loading, multi-process data loading
        drop_last=HP["drop_last"],            # drop last batch if size is less than batch_size
        pin_memory=HP["pin_memory"]           # faster data transfer to GPU
    )
    
    return data_loader
from torch.utils import data
import torch 

class ATISData(data.Dataset):
    def __init__(self, X, y,y2):
        self.len = len(X)
        self.x_data = X
        self.y_data = y
        self.y_data_2 = y2

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index],self.y_data_2[index]
             
    def __len__(self):
        return self.len
from torch.utils.data import Dataset
import pandas as pd

class CustomDataset(Dataset):
    
    def __init__(self, csv: str):
        self.df = pd.read_csv(csv)
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        # TODO: GET ITEM: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
        return self.df.loc[index]
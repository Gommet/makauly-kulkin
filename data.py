from torch.utils.data import Dataset
import pandas as pd
from IPython.display import display
from torch import Tensor


class CustomDataset(Dataset):
    
    def __init__(self, pickle: str, group: str, shift: int, train_ratio: float, is_train: bool):
        df = pd.read_pickle(pickle)
        df['new_sales'] = df.groupby(level=0)['sales'].shift(-shift)
        
        df = df.dropna()
        self.df = df.loc[group].reset_index(drop=True)
        train_size = int(train_ratio * self.df.shape[0])
        self.df = self.df.iloc[:train_size] if is_train else self.df.iloc[train_size:].reset_index(drop=True)
        
    def __len__(self):
        return len(self.df) - 1
    
    def __getitem__(self, idx):
        # TODO: GET ITEM: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
        iloc = self.df.iloc(axis=1)
        X, y = iloc[:-1], iloc[-1]
        return Tensor(X.loc[idx]), Tensor([y.loc[idx]])

class MoreShiftCustomDataset(Dataset):
    
    def __init__(self, pickle: str, group: str, shift: int, train_ratio: float, is_train: bool):
        df = pd.read_pickle(pickle)
        group_sales = df.groupby(level=0)['sales']
        df['new_sales'] = group_sales.shift(-shift)
        df['sales4'] = group_sales.shift(4)
        df['sales7'] = group_sales.shift(7)
        df['sales10'] = group_sales.shift(10)
        df['sales14'] = group_sales.shift(14)
        df = df.dropna()
        self.df = df.loc[group].reset_index(drop=True)
        train_size = int(train_ratio * self.df.shape[0])
        self.df = self.df.iloc[:train_size] if is_train else self.df.iloc[train_size:].reset_index(drop=True)
        
    def __len__(self):
        return len(self.df) - 1
    
    def __getitem__(self, idx):
        # TODO: GET ITEM: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
        iloc = self.df.iloc(axis=1)
        X, y = iloc[:-1], iloc[-1]
        return Tensor(X.loc[idx]), Tensor([y.loc[idx]])

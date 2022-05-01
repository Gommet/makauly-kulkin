
import data
import model as model_mod
from torch import nn
from torch.utils.data import DataLoader
import torch
from torch.utils.tensorboard import SummaryWriter
import functools as func
from sklearn.metrics import r2_score
import seaborn as sns
import sys
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

group = sys.argv[1]
NUMBER = 11

dataset = data.MoreShiftCustomDataset('clean.pickle', group, 7, 0.8, True)
dataset_test = data.MoreShiftCustomDataset('clean.pickle', group, 7, 0.8, False)

train_loader = DataLoader(dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False)

writer = SummaryWriter(f"tensorb/{group}{NUMBER}")
flatten = lambda y : list(map(lambda x: x, func.reduce(lambda x, y: list(x) + list(y), y)))

model = model_mod.ShiftRNN(6, 10, 2)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
start_epoch = 0
loss_final = float("inf")
try:
    checkpoint = torch.load(f'models/{NUMBER}/model-{group}.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    loss_final = checkpoint['loss']
except FileNotFoundError:
    start_epoch = 0
    loss_final = float("inf")
epochs = 50000
for e in range(start_epoch, start_epoch + epochs):
    model.train()
    
    preds_train = []
    real_train = []
    hidden = None
    for x, y in train_loader:
        pred, hidden = model(x, hidden)
        hidden = hidden.data
        preds_train.extend(pred.detach().numpy())
        real_train.extend(y.detach().numpy())
        loss_train = criterion(pred, y)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
    hidden = None
    model.eval()
    res = {
    'Real':flatten(real_train),
    'Output': flatten(preds_train),
    }
    plt.figure(figsize=(16, 9))
    sns.lineplot(data=res)
    output_fig = f"img/train{NUMBER}/{e}.png"
    plt.savefig(output_fig)
    
    print(f"Loss epoch {e}: ", loss_train.item())    
    if loss_final > loss_train.item():
        loss_final = loss_train.item()
        print("Saving...")
        torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_train,
            }, f'models/{NUMBER}/model-{group}.pth')

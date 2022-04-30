from torch import nn
from torch import optim

from model import RNN

def main():
    model = RNN(input_size=1, hidden_dim=10, n_layers=3)
    
    criterion = nn.MSELoss()
    optimiser = optim.LBFGS(model.parameters(), lr=0.08)
    
    

if __name__ == '__main__':
    main()
    
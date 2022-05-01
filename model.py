from torch import nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, input_size, hidden_dim, n_layers):
        super(RNN, self).__init__()
        
        self.hidden_dim=hidden_dim

        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, hidden):
        # x (batch_size, seq_length, input_size)
        # hidden (n_layers, batch_size, hidden_dim)
        # r_out (batch_size, time_step, hidden_size)
        # get RNN outputs
        r_out, hidden = self.rnn(x, hidden)
        # shape output to be (batch_size*seq_length, hidden_dim)
        r_out = r_out.view(-1, self.hidden_dim)  
        
        output = self.fc(r_out)
        
        return output, hidden


class RNN2(nn.Module):
    def __init__(self, input_size, hidden_dim, n_layers):
        super(RNN2, self).__init__()
        
        self.hidden_dim=hidden_dim

        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        
        self.fc1 = nn.Linear(hidden_dim, 20)
        self.output = nn.Linear(20, 1)

    def forward(self, x, hidden):
        # x (batch_size, seq_length, input_size)
        # hidden (n_layers, batch_size, hidden_dim)
        # r_out (batch_size, time_step, hidden_size)
        # get RNN outputs
        r_out, hidden = self.rnn(x, hidden)
        # shape output to be (batch_size*seq_length, hidden_dim)
        r_out = r_out.view(-1, self.hidden_dim)  
        
        x = F.relu(self.fc1(r_out))
        
        output = self.output(x)
        
        return output, hidden
    
class ShiftRNN(nn.Module):
    def __init__(self, input_size, hidden_dim, n_layers):
        super(ShiftRNN, self).__init__()
        
        self.hidden_dim=hidden_dim

        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        
        self.fc1 = nn.Linear(hidden_dim, 20)
        self.fc2 = nn.Linear(20, 32)
        self.fc3 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)

    def forward(self, x, hidden):
        # x (batch_size, seq_length, input_size)
        # hidden (n_layers, batch_size, hidden_dim)
        # r_out (batch_size, time_step, hidden_size)
        # get RNN outputs
        r_out, hidden = self.rnn(x, hidden)
        # shape output to be (batch_size*seq_length, hidden_dim)
        r_out = r_out.view(-1, self.hidden_dim)  
        
        x = F.relu(self.fc1(r_out))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        output = self.output(x)
        
        return output, hidden
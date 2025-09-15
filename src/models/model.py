from torch import nn




class SimpleModel(nn.Module):
    def __init__(self, input_dim):
        super(SimpleModel, self).__init__()
        
        self.input = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 16)
        self.fc5 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)  # Dropout with probability 0.5

    def forward(self, x):
        x = self.relu(self.input(x))
        # x = self.dropout(x)  # Apply dropout
        x = self.relu(self.fc2(x))
        # x = self.dropout(x)  # Apply dropout
        x = self.relu(self.fc3(x))
        # x = self.dropout(x)  # Apply dropout
        x = self.relu(self.fc4(x))
        # x = self.dropout(x)  # Apply dropout
        
        x = self.fc5(x)

        return x

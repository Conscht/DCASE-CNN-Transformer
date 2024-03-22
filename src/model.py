import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(nn.Module):
    def __init__(self, hidden_dim=128, num_classes=15):
        super().__init__()
        self.l1 = nn.Linear(200, 50)
        self.l2 = nn.Linear(50, 50)
        self.l3 = nn.Linear(50, num_classes)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # Linear Layers
        x = self.l1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.l2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.l3(x)
        x = F.relu(x) # 32, 274, 15
        
        # Mean
        x = x.mean(dim=1)

        return x
    
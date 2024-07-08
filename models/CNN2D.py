from torch import nn
from torch.nn import functional as F
import torch

class Encoder_CNN2D(nn.Module):
    def __init__(self,p=0.4):
        super(Encoder_CNN2D, self).__init__()
        self.basic_model = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(p),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, (3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(p),
            nn.MaxPool2d(2, 2)
        )

        self.header = nn.Sequential(
            nn.Linear(32 * 8*7, 128,bias=False),
        )
    def forward(self,x):
        x = F.one_hot(x.to(torch.int64), 30).float().view(x.size(0), -1)
        x = x.view(-1, 1, 32, 30)
        x = self.basic_model(x)
        x = x.view(-1, 32 * 8*7)
        x = self.header(x)
        return x

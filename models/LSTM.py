import torch
import torch.nn as nn
from torch.nn import functional as F

# 定义LSTM模型
class Encoder_LSTM(nn.Module):
    def __init__(self, input_size=30, hidden_size=64, num_layers=2, bidirectional=True,out_dim=128):
        super(Encoder_LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        # 定义LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            bidirectional = bidirectional)

        # 定义全连接层
        self.fc = nn.Linear(hidden_size*2, out_dim,bias=False)

    def forward(self, x):
        # 初始化隐藏状态
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # 前向传播
        x = F.one_hot(x.to(torch.int64), 30).float().view(x.size(0), -1)
        x = x.view(-1, 32, 30)
        out, _ = self.lstm(x)
        # 提取最后一个时间步的输出作为模型的输出
        out = self.fc(out[:, -1, :])
        return out

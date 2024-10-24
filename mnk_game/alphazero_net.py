import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleBackbone(nn.Module):
    def __init__(self, padding):
        super(SimpleBackbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=padding)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=padding)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init._normal_(m.weight, 0, 0.01)
                nn.init._constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        return x


class AlphaZeroNet(nn.Module):
    def __init__(self, m, n, k, backbone="simple"):
        super(AlphaZeroNetwork, self).__init__()
        # handle near-edge cases
        padding = math.ceil(k / 2)
        if backbone == "simple":
            self.backbone = SimpleBackbone(padding)
            self.backbone.init_weights()
        else:
            raise NotImplementedError
        self.fc_policy = nn.Linear(128 * m * n, m * n)
        self.fc_value = nn.Linear(128 * board_size * board_size, 1)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # Flatten
        policy = torch.softmax(self.fc_policy(x), dim=-1)
        value = torch.tanh(self.fc_value(x))
        return policy, value


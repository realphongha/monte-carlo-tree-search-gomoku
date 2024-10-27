import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def calc_conv2d_output(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """takes a tuple of (h,w) and returns a tuple of (h,w)"""

    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, kernel_size)
    h = math.floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
    w = math.floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
    return h, w


class SimpleBackbone(nn.Module):
    def __init__(self, padding, num_filters=128):
        super(SimpleBackbone, self).__init__()
        self.padding = padding
        self.conv1 = nn.Conv2d(2, num_filters, kernel_size=3, padding=self.padding)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=self.padding)

    def get_output_size(self, h_w):
        h_w = calc_conv2d_output(h_w, kernel_size=3, pad=self.padding)
        h_w = calc_conv2d_output(h_w, kernel_size=3, pad=self.padding)
        return h_w

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        return x


class AlphaZeroNet(nn.Module):
    def __init__(self, m, n, k, backbone="simple"):
        super(AlphaZeroNet, self).__init__()
        # handle near-edge cases
        self.padding = math.ceil(k / 2)
        self.num_filters = 128
        if backbone == "simple":
            self.backbone = SimpleBackbone(self.padding, self.num_filters)
            self.backbone.init_weights()
            h_w = self.backbone.get_output_size((m, n))
            h, w = calc_conv2d_output(h_w, kernel_size=3, pad=0)
            self.output_size = h * w
        else:
            raise NotImplementedError

        self.policy = nn.Sequential(
            nn.Conv2d(self.num_filters, 2, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * self.output_size, m * n),
            nn.Softmax(dim=-1)
        )

        self.value = nn.Sequential(
            nn.Conv2d(self.num_filters, 1, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.output_size, 1),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.backbone(x)
        policy = self.policy(x)
        value = self.value(x)
        return policy, value


if __name__ == "__main__":
    import numpy as np
    from board_state.board import to_board, to_bitboard

    board = [[-1, 1 , -1, 0 ],
             [0 , -1, 0 , -1],
             [-1, 1 , 0 , 0 ],
             [1 , -1, 0 , 0 ]]
    board = np.array(board).astype(np.float32)
    m = board.shape[0]
    n = board.shape[1]
    k = 3
    board = to_board(to_bitboard(board), m, n)
    inp = torch.tensor(board).float().unsqueeze(0)
    net = AlphaZeroNet(m, n, 3)
    print(net)
    policy, value = net(inp)
    print(policy)
    print(value)


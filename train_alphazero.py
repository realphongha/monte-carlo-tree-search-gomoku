import argparse
import yaml
import sys
import time
import os
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from mnk_game.alphazero_mnkgame import AlphaZeroMnkGame
from mnk_game.mcts_mnkgame import MonteCarloTreeSearchMnkGame
from mnk_game.alphazero_net import AlphaZeroNet
from board_state.mnk_board import MnkBoard


def self_play(cfg, net=None):
    num_games = cfg["bot"]["alphazero"]["self_play_games"]
    print("Self-playing %d games..." % num_games)
    m, n, k = cfg["board_game"]["m"], cfg["board_game"]["n"], cfg["board_game"]["k"]
    training_data = []
    for _ in tqdm(range(num_games)):
        game = AlphaZeroMnkGame(m, n, k, **cfg["bot"]["alphazero"])
        game.init_net(net)
        board = MnkBoard(m, n, k)
        possible_pos = board.get_possible_pos()
        # -1, 1 - turns (magic numbers)
        turn = random.choice((-1, 1))
        moves = []
        res = 0
        game_data = []
        while not res and possible_pos:
            move, policy = game.solve(board, turn, moves)
            bb = game.bitboard_to_tensor(board.get_board())[0]
            game_data.append((bb, turn, policy))
            assert move in possible_pos
            board.put(turn, move)
            moves.append(move)
            possible_pos = board.get_possible_pos()
            turn = -turn
            res = board.check_endgame()

        for board, turn, policy in game_data:
            training_data.append((board, turn, policy, res))
    return training_data


class MnkDataset(torch.utils.data.Dataset):
    def __init__(self, data, m, n):
        self.data = data
        self.m, self.n = m, n

    def __len__(self):
        return len(self.data)

    def policy_to_2d(self, policy):
        policy_2d = np.zeros((self.m, self.n))
        for i in range(self.m):
            for j in range(self.n):
                policy_2d[i, j] = policy[i * self.n + j]
        return policy_2d

    def policy_to_1d(self, policy):
        policy_1d = np.zeros((self.m * self.n,))
        for i in range(self.m):
            for j in range(self.n):
                policy_1d[i * self.n + j] = policy[i, j]
        return policy_1d

    def __getitem__(self, idx):
        board, turn, policy, res = self.data[idx]
        board = board.clone().detach().cpu().numpy()
        policy = self.policy_to_2d(policy)
        # align board presentation for both turn (1 and -1)
        board *= turn
        # flip left-right augmentation
        if random.random() < 0.5:
            board = np.flip(board, axis=2)
            policy = np.flip(policy, axis=1)
        # flip up-down augmentation
            board = np.flip(board, axis=1)
            policy = np.flip(policy, axis=0)
        policy = self.policy_to_1d(policy)
        board = torch.tensor(board.copy(), dtype=torch.float)
        policy = torch.tensor(policy, dtype=torch.float)

        return board, policy, res


def train(data, cfg):
    m, n, k = cfg["board_game"]["m"], cfg["board_game"]["n"], cfg["board_game"]["k"]
    device = cfg["bot"]["alphazero"]["device"]
    eps = cfg["bot"]["alphazero"]["eps"]
    print("Training the network for %i epochs..." % eps)
    dataset = MnkDataset(data, m, n)

    loader = DataLoader(dataset,
                              batch_size=16,
                              shuffle=True,
                              num_workers=2)

    net = AlphaZeroNet(m, n, k)
    net.init_weights()
    net.to(device)
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    policy_loss = torch.nn.CrossEntropyLoss()
    value_loss = torch.nn.MSELoss()

    for epoch in tqdm(range(eps)):
        for board, policy, value in loader:
            optimizer.zero_grad()
            board = board.to(device).float()
            policy = policy.to(device).float()
            value = value.to(device).float()
            policy_pred, value_pred = net(board)
            policy_loss_ = policy_loss(policy_pred, policy)
            value_loss_ = value_loss(value_pred.view(-1), value)
            loss = policy_loss_ + value_loss_
            loss.backward()
            optimizer.step()

    return net


def play(player1, player2, cfg):
    m, n, k = cfg["board_game"]["m"], cfg["board_game"]["n"], cfg["board_game"]["k"]
    board = MnkBoard(m, n, k)
    possible_pos = board.get_possible_pos()
    moves = []
    while possible_pos:
        move = player1.predict(board, 1, moves)

        assert move in possible_pos
        board.put(1, move)
        moves.append(move)
        res = board.check_endgame()
        if res:
            break
        possible_pos = board.get_possible_pos()

        move = player2.predict(board, -1, moves)
        assert move in possible_pos
        board.put(-1, move)
        moves.append(move)
        res = board.check_endgame()
        if res:
            break
        possible_pos = board.get_possible_pos()
    return res


def arena(last_net, new_net, cfg):
    m, n, k = cfg["board_game"]["m"], cfg["board_game"]["n"], cfg["board_game"]["k"]
    games = cfg["bot"]["alphazero"]["arena_games"]
    new_game = AlphaZeroMnkGame(m, n, k, **cfg["bot"]["alphazero"])
    new_game.init_net(new_net)
    print("Arena: ")
    print("Versus last iteration's net:")
    evolved = True
    if last_net is not None:
        last_game = AlphaZeroMnkGame(m, n, k, **cfg["bot"]["alphazero"])
        last_game.init_net(last_net)
        total = []
        for _ in tqdm(range(games)):
            res = play(new_game, last_game, cfg)
            total.append(res == 1)
            res = play(last_game, new_game, cfg)
            total.append(res == -1)
        percent = sum(total) / len(total) * 100
        print("Winrate against last iteration: %.2f%%" % percent)
        evolved = percent > 50.0

    print("Versus pure MCTS:")
    mcts = MonteCarloTreeSearchMnkGame(**cfg["bot"]["mcts"])
    mcts.debug = False
    total = []
    for _ in tqdm(range(games)):
        res = play(new_game, mcts, cfg)
        total.append(res == 1)
        res = play(mcts, new_game, cfg)
        total.append(res == -1)
    percent = sum(total) / len(total) * 100
    print("Winrate against pure MCTS: %.2f%%" % percent)
    return evolved


def main(cfg):
    last_net = None
    m, n, k = cfg["board_game"]["m"], cfg["board_game"]["n"], cfg["board_game"]["k"]
    date_time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join(cfg["bot"]["alphazero"]["exp_dir"], date_time_str)
    os.makedirs(exp_dir)
    for it in range(cfg["bot"]["alphazero"]["it"]):
        print(f"Iteration {it+1}:")
        training_data = self_play(cfg, last_net)
        net = train(training_data, cfg)
        evolved = arena(last_net, net, cfg)

        # save net
        torch.save(net.state_dict(), os.path.join(exp_dir, f"it{it}.pth"))
        if evolved or last_net is None:
            last_net = net
            torch.save(net.state_dict(), os.path.join(exp_dir, f"best.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/tic_tac_toe.yaml',
                        help='path to config file')
    opt = parser.parse_args()
    with open(opt.cfg, "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit()
    main(cfg)


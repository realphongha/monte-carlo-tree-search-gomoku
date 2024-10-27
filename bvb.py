import argparse
import yaml
import sys
from gui.mnk_gui import MnkGUI
from mnk_game.mcts_mnkgame import MonteCarloTreeSearchMnkGame


def main(cfg):
    gui = MnkGUI(**cfg["board_game"])
    gui.player1 = MonteCarloTreeSearchMnkGame(**cfg["bot"]["mcts"])
    gui.player2 = MonteCarloTreeSearchMnkGame(**cfg["bot"]["mcts"])
    gui.main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',
                        type=str,
                        default='configs/gomoku.yaml',
                        help='path to config file')
    opt = parser.parse_args()
    with open(opt.cfg, "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit()
    main(cfg)

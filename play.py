import argparse
from calendar import c
import yaml
from mnk_game.game import Game
from mnk_game.mcts_mnkgame import MonteCarloTreeSearchMnkGame


def main(cfg):
    game = Game(**cfg["board_game"])
    bot_alg = cfg["bot"]["algorithm"]
    if bot_alg == "mcts":
        bot = MonteCarloTreeSearchMnkGame(**cfg["bot"]["config"])
    else:
        raise NotImplementedError("%s algorithm is not implemented!" % bot_alg)
    game.add_bot(bot)
    game.main()


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
            quit()
    main(cfg)

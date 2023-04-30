import argparse
import yaml
import sys
from mnk_game.game import Game


def main(cfg):
    game = Game(**cfg["board_game"])
    game.set_bot(cfg["bot"]["algorithm"])
    game.add_bot_config(cfg["bot"]["config"])
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
            sys.exit()
    main(cfg)

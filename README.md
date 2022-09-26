# mnk-game-bot
AI bot to play [m,n,k-games](https://en.wikipedia.org/wiki/M,n,k-game) (Tic-tac-toe, Gomoku...) with simple Pygame GUI for Player vs. Bot.

# Implemented stuffs
## Bot algorithms
* Monte Carlo tree search
## Game configs
* Tic-tac-toe
* Gomoku 7x7
* Gomoku 15x15
## Cython optimization
* For end-game board checking
* For getting possible moves on board
* For calculating UCB + score (not so big improvement)
## Parallelism
* Single-run parallelization:
```
@inproceedings{cazenave2007parallelization,
  title={On the parallelization of UCT},
  author={Cazenave, Tristan and Jouandeau, Nicolas},
  booktitle={Computer games workshop},
  year={2007}
}
```

# How to install
First `git clone https://github.com/realphongha/mnk-game-bot.git` to clone this repo.  
Go inside the repo: `cd mnk-game-bot`

## Install requirements
Run:
```
pip install -r requirements.txt
```

## Compile Cython file
Run:
```
python setup.py build_ext --inplace 
```

# How to run
```
python play.py --cfg path/to/config/file.yaml
```


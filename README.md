# mnk-game-bot
AI bot to play [m,n,k-games](https://en.wikipedia.org/wiki/M,n,k-game) (Tic-tac-toe, Gomoku...) with simple Pygame GUI for Player vs. Bot.

# Implemented stuffs
## Bot algorithms
* Monte Carlo tree search
## Game configs
* Gomoku
* Tic-tac-toe
## Cython optimization
* For end-game board checking
## Parallelism
* to be added soon...

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


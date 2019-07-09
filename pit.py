import Arena
from MCTS import MCTS
from vikingdoms.VikingdomsGame import VikingdomsGame
from vikingdoms.VikingdomsPlayers import *
from vikingdoms.keras.NNet import NNetWrapper as NNet

import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

g = VikingdomsGame(4)

# all players
rp = RandomPlayer(g).play
gp = GreedyVikingdomsPlayer(g).play
hp = HumanVikingdomsPlayer(g).play

# nnet players
n1 = NNet(g)
n1.load_checkpoint('./pretrained_models/vikings/','best.pth.tar')
args1 = dotdict({'numMCTSSims': 150, 'maxMCTSMoveDepth': 32, 'cpuct':1.0, 'epsilon': 0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))


#n2 = NNet(g)
#n2.load_checkpoint('/dev/8x50x25/','best.pth.tar')
#args2 = dotdict({'numMCTSSims': 25, 'cpuct':1.0})
#mcts2 = MCTS(g, n2, args2)
#n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

# play against human
#arena = Arena.Arena(n1p, hp, g, display=Board.display)
#print(arena.playGames(2, verbose=True))

# play against bot
arena = Arena.Arena(n1p, rp, g, display=None)
print(arena.playGames(20, verbose=False))

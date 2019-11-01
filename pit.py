import Arena
from MCTS import MCTS
from kamisado.KamisadoGame import KamisadoGame as Game
#from kamisado.KamisadoGame import display
from kamisado.KamisadoPlayers import *
from kamisado.keras.NNet import NNetWrapper as NNet

import numpy as np
from utils import dotdict

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

g = Game(6)

# all players
rp = RandomPlayer(g).play
#gp = GreedyPlayer(g).play
#hp = HumanPlayer(g).play

# nnet players
n1 = NNet(g)
n1.load_checkpoint(folder='./pretrained_models/kamisado', filename='6x100x150_best.pth.tar')
args1 = dotdict({'numMCTSSims': 150, 'cpuct': 1.0, 'maxMCTSMoveDepth': 32, 'epsilon': 0, 'dirAlpha': 0})
mcts1 = MCTS(g, n1, args1)

def createPlayFromMCTS(mcts):
    def play(canonicalBoard : Board):
        return np.argmax(mcts.getActionProb(canonicalBoard, temp=0))
    return play


n1p = createPlayFromMCTS(mcts1)

n2 = NNet(g)
n2.load_checkpoint(folder='./pretrained_models/kamisado', filename='checkpoint_23.pth.tar')
#args2 = dotdict({'numMCTSSims': 25, 'cpuct':1.0})
mcts2 = MCTS(g, n2, args1)
n2p = createPlayFromMCTS(mcts2)

# play against human
#arena = Arena.Arena(n1p, hp, g, display=display)
#print(arena.playGames(2, verbose=True))

# play against bot
#arena = Arena.Arena(n1p, rp, g, display=None)
arena = Arena.Arena(n1p, n2p, g, display=None)
print(arena.playGames(20, verbose=False))

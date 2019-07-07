from Coach import Coach
#from othello.OthelloGame import OthelloGame as Game
#from othello.pytorch.NNet import NNetWrapper as nn
#from othello.keras.NNet import NNetWrapper as nn
from vikingdoms.VikingdomsGame import VikingdomsGame as Game
from vikingdoms.keras.NNet import NNetWrapper as nn
from utils import *
import cProfile, pstats

args = dotdict({
    'numIters': 100,
    'numEps': 100,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 25,
    'arenaCompare': 40,
    'cpuct': 1,
    #'epsilon': 0.25,
    #'dirAlpha': (10.0/35)/0.4,  # (want 10 moves / 35 valid moves average) / 0.4 move packing efficiency
    #'dirAlpha': (10.0/35),  # (want 10 moves / 35 valid moves average)
    'epsilon': 0,
    'dirAlpha': 0,
    'maxMCTSMoveDepth': 32,
    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('temp','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})

if __name__=="__main__":
    pr = cProfile.Profile()
    try:
        #pr.enable()

        g = Game(4)
        nnet = nn(g)

        if args.load_model:
            nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

        c = Coach(g, nnet, args)
        if args.load_model:
            print("Load trainExamples from file")
            c.loadTrainExamples()
        c.learn()
    finally:
        pass
        #pr.disable()
        #ps = pstats.Stats(pr).sort_stats('cumulative')
        #ps.print_stats(50)

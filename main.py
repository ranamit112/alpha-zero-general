from Coach import Coach
from kamisado.KamisadoGame import KamisadoGame as Game
from kamisado.keras.NNet import NNetWrapper as nn
from utils import dotdict
import cProfile, pstats

args = dotdict({
    'numIters': 1000,
    'numEps': 100,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 150,
    'arenaCompare': 40,
    'cpuct': 1,
    'epsilon': 0,
    'dirAlpha': 0,
    'maxMCTSMoveDepth': 32,
    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('temp','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})

ENABLE_PROFILE = False

if __name__ == "__main__":
    pr = cProfile.Profile()
    try:
        if ENABLE_PROFILE:
            pr.enable()

        g = Game(6)
        nnet = nn(g)

        if args.load_model:
            nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

        c = Coach(g, nnet, args)
        if args.load_model:
            print("Load trainExamples from file")
            c.loadTrainExamples()
        c.learn()
    finally:
        if ENABLE_PROFILE:
            pr.disable()
            ps = pstats.Stats(pr).sort_stats('cumulative')
            ps.print_stats(50)

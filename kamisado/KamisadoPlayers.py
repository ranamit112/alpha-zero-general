import numpy as np
import math
from .VikingdomsGame import VikingdomsGame, Board, BoardNumpyUtil
from .VikingdomsLogic import DisplayRowNotation

class RandomPlayer:
    def __init__(self, game: VikingdomsGame):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)
        return np.random.choice(np.flatnonzero(valids))


class HumanVikingdomsPlayer:
    def __init__(self, game: VikingdomsGame):
        self.game = game

    def play(self, board):
        # display(board)
        n = self.game.n
        valids = self.game.getValidMoves(board, 1)
        assert len(valids) == self.game.getActionSize()
        assert len(valids) == n**2 + (n**4)*5 + 1
        for i in range(len(valids)):
            if valids[i]:
                move = BoardNumpyUtil.move_from_numpy_val(self.game.n, i)
                move_height, (x, y), (x2, y2) = move
                if i == len(valids) - 1:
                    print("{}: Skip".format(i))
                elif i < n**2:
                    print("{}: Place piece in {}".format(
                        i,
                        DisplayRowNotation.get_position(self.game.n, x2, y2)))
                else:
                    print("{}: Move {} pieces from {} to {}".format(
                        i, move_height,
                        DisplayRowNotation.get_position(self.game.n, x, y),
                        DisplayRowNotation.get_position(self.game.n, x2, y2)))
        while True:
            print("Enter selection: ", end='')
            inp = input()
            action = int(inp)
            #x,y = [int(x) for x in a.split(' ')]
            #a = self.game.n * x + y if x!= -1 else self.game.n ** 2
            if valids[action]:
                break
            else:
                print('Invalid')

        return action


class GreedyVikingdomsPlayer:
    def __init__(self, game: VikingdomsGame):
        self.game = game

    def play(self, board: Board):
        valids = self.game.getValidMoves(board, 1)
        candidates = np.full(len(valids), board.n*board.n*board.WIN_TOWER*(-10.0))
        for a in np.flatnonzero(valids):
            next_board, _ = self.game.getNextState(board, 1, a)
            score = self.getScore(next_board, 1)
            candidates[a] = score
        best_act = np.random.choice(np.flatnonzero(candidates == candidates.max()))
        return best_act

    def getScore(self, board: Board, color: int) -> float:
        K1 = 1
        K2 = 0.75
        s = 0.0
        for x in range(board.n):
            for y in range(board.n):
                h = board.get_height_of_tower(x, y)
                vc = board.get_top_piece_player(x, y)
                if h == board.WIN_TOWER:
                    # winning is Infinity, losing -Inf
                    return math.inf * vc * color
                s += h*K1 * vc * color
        s += board.pieces_left[0] * K2 * color  # add points for other player left pieces
        s -= board.pieces_left[1] * K2 * color  # remove points for your player
        return s

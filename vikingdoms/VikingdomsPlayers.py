import numpy as np
from .VikingdomsGame import Game, Board, BoardNumpyUtil

class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)
        valids_pos = list()
        for i in range(len(valids)):
            if valids[i]:
                valids_pos.append(i)

        a = np.random.randint(len(valids_pos))
        return valids_pos[a]


class HumanVikingdomsPlayer():
    def __init__(self, game):
        # type: (Game) -> None
        self.game = game

    def play(self, board):
        # display(board)
        n = self.game.n
        valids = self.game.getValidMoves(board, 1)
        assert len(valids) == self.game.getActionSize()
        assert len(valids) == n**2 + (n**4)*5 + 1
        for i in range(n**2 + n**2 + (n**4)*5):
            if valids[i]:
                move = BoardNumpyUtil.move_from_numpy_val(self.game.n, valids[i])
                print("{}: Move {} pieces from {},{} to {},{}".format(
                    i, move[0], move[1][0], move[1][1], move[2][0], move[2][1]))
#        for i in range(n**2, n**2 + (n**4)*5):
#            if valids[i]:
#                move = BoardNumpyUtil.move_from_numpy_val(self.game.n, valids[i])
#                print("{}: Move {} pieces from {},{} to {},{}".format(
#                    i, move[0], move[1][0], move[1][1], move[2][0], move[2][1]))
        if valids[len(valids)-1]:
            print("{}: Skip".format(len(valids)-1))
        while True:
            inp = input()
            action = int(inp)
            #x,y = [int(x) for x in a.split(' ')]
            #a = self.game.n * x + y if x!= -1 else self.game.n ** 2
            if valids[action]:
                break
            else:
                print('Invalid')

        return action


class GreedyVikingdomsPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)
        candidates = []
        for a in range(self.game.getActionSize()):
            if valids[a] != 1:
                continue
            next_board, _ = self.game.getNextState(board, 1, a)
            score = self.getScore(next_board, 1)
            candidates += [(-score, a)]
        candidates.sort()
        return candidates[0][1]

    def getScore(self, board, color):
        # type: (np.ndarray, int) -> int
        K1 = 1
        K2 = 1
        s = 0
        n = board.shape[0]
        for x in range(n):
            for y in range(n):
                v = board[x][y]
                h = Board.get_height_of_tower_val(v)
                vc = Board.get_top_piece_player_val(v)
                s += h*K1 * vc * color
        s += (board[n][0]) * K2 * color  # add points for other player left pieces
        s -= (board[n][1]) * K2 * color  # remove points your player
        return s

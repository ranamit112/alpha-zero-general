from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .VikingdomsLogic import Board, BoardNumpyUtil
import numpy as np


class VikingdomsGame(Game):
    """
    Implement Vikingdoms (https://boardgamegeek.com/boardgame/206780/vikingdoms)
    First Phase - Only Tactic game, 1 win to win
    Second Phase - Only Tactic game, Best 3 out of 5 to win
    Third Phase - Tactic game + Strategy Game
    """

    def __init__(self, n=4):
        super().__init__()
        self.n = n

    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.n)
        return BoardNumpyUtil.board_to_numpy(b)

    def getBoardSize(self):
        # (a,b) tuple
        return BoardNumpyUtil.board_shape_numpy(self.n)

    def getActionSize(self):
        # type: () -> int
        # return number of actions
        # adding piece to board, move tower part from one point of board to another times height, skip
        return (self.n * self.n) + (self.n * self.n)*(self.n * self.n)*(Board.WIN_TOWER-1) + 1

    def getNextState(self, board, player, action):
        # type: (np.ndarray, int, int) -> (np.ndarray, int)
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        assert action <= self.getActionSize()-1
        # skip
        if action == self.getActionSize()-1:
            return (board, 1 - player)
        b = BoardNumpyUtil.board_from_numpy(board)
        move = BoardNumpyUtil.move_from_numpy_val(self.n, action)
        b.execute_move(move, player)
        return (BoardNumpyUtil.board_to_numpy(b), 1 - player)

    # modified
    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        b = BoardNumpyUtil.board_from_numpy(board)
        valids = [0] * self.getActionSize()
        legal_moves = b.get_legal_moves(player)
        if len(legal_moves) == 0:  # last position is skip
            valids[-1] = 1
            return np.array(valids)

        for m in legal_moves:
            valids[BoardNumpyUtil.move_to_numpy_val(self.n, m)] = 1
        return np.array(valids)

    # modified
    def getGameEnded(self, board, player):
        # type: (np.ndarray, int) -> int or float
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = BoardNumpyUtil.board_from_numpy(board)
        for x in range(self.n):
            for y in range(self.n):
                if b.get_height_of_tower(x, y) == b.WIN_TOWER:
                    return b.get_top_piece_player(x, y)
        if b.has_legal_moves():
            return 0
        return 1e-4  # TODO: I don't understand this

    def getCanonicalForm(self, board, player):
        # type: (np.ndarray, int) -> np.ndarray
        # return state if player==1, else return -state if player==-1
        return BoardNumpyUtil.get_canonical_board(self.n, board, player)

    # modified
    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert(len(pi) == self.getActionSize())  # -1 for pass

        # board shape is [n,n+1]
        # pi is one dimensional of n*n + (n**4)*5 + 1
        # for the sake of this part we will treat it as a 5 dimensional array of sizes n,n,n,n,5 + the skip element
        pi_board1 = np.reshape(pi[0:self.n**2], (self.n, self.n))
        pi_board2 = np.reshape(pi[self.n**2:-1], (self.n, self.n, self.n, self.n, 5))
        pi_skip = pi[-1]
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi1 = np.rot90(pi_board1, k=i, axes=(0, 1))
                newPi2 = np.rot90(pi_board2, k=i, axes=(0, 1))
                newPi2 = np.rot90(newPi2,    k=i, axes=(2, 3))
                if j:
                    newB = np.fliplr(newB)
                    newPi1 = np.fliplr(newPi1)
                    newPi2 = np.flip(newPi2, 1)
                    newPi2 = np.flip(newPi2, 3)
                l += [(newB, list(newPi1.ravel()) + list(newPi2.ravel()) + [pi_skip])]
        return l

    def stringRepresentation(self, board):
        # 5x4 (or n+1 * n) numpy array (canonical board)
        return board.tostring()

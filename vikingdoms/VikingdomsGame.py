from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .VikingdomsLogic import Board, BoardNumpyUtil
import numpy as np
import copy

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

    def getInitBoard(self) -> Board:
        # return initial board
        b = Board(self.n)
        return b

    def getBoardSize(self):
        # (a,b) tuple
        return BoardNumpyUtil.board_shape_numpy(self.n)

    def getActionSize(self) -> int:
        # return number of actions
        # adding piece to board, move tower part from one point of board to another times height, skip
        return (self.n * self.n) + (self.n * self.n)*(self.n * self.n)*(Board.WIN_TOWER-1) + 1

    def getNextState(self, board: Board, player: int, action: int) -> (Board, int):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        assert action <= self.getActionSize()-1
        # skip
        if action == self.getActionSize()-1:
            return (board, 1 - player)
        move = BoardNumpyUtil.move_from_numpy_val(self.n, action)
        board = copy.copy(board)
        board.execute_move(move, player)
        return board, 1 - player

    # modified
    def getValidMoves(self, board: Board, player: int) -> np.ndarray:
        # return a fixed size binary vector
        valids = [0] * self.getActionSize()
        legal_moves = board.get_legal_moves(player)
        if len(legal_moves) == 0:  # last position is skip
            valids[-1] = 1
            return np.array(valids)

        for m in legal_moves:
            valids[BoardNumpyUtil.move_to_numpy_val(self.n, m)] = 1
        return np.array(valids)

    # modified
    def getGameEnded(self, board: Board, player: int) -> int or float:
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        for x in range(self.n):
            for y in range(self.n):
                if board.get_height_of_tower(x, y) == board.WIN_TOWER:
                    won_color = board.get_top_piece_player(x, y)
                    # print("player {} won".format(won_color))
                    return won_color
        if board.has_legal_moves():
            return 0
        return 1e-4  # TODO: I don't understand this

    def getCanonicalForm(self, board: Board, player: int) -> Board:
        if player == 1:
            return board
        # return state if player==1, else return -state if player==-1
        b = BoardNumpyUtil.board_to_numpy(board)
        bc = BoardNumpyUtil.get_canonical_board(self.n, b, player)
        board2 = BoardNumpyUtil.board_from_numpy(bc)
        board2.last_move = board.last_move
        return board2

    # modified
    def getSymmetries(self, board: Board, pi: np.ndarray) -> list:
        # mirror, rotational
        assert(len(pi) == self.getActionSize())  # -1 for pass

        # board shape is [n,n]
        # pi is one dimensional of n*n + (n**4)*5 + 1
        # for the sake of this part we will treat it as a 5 dimensional array of sizes n,n,n,n,5 + the skip element
        pi_board1 = np.reshape(pi[0:self.n**2], (self.n, self.n))
        pi_board2 = np.reshape(pi[self.n**2:-1], (self.n, self.n, self.n, self.n, 5))
        pi_skip = pi[-1]
        l = []
        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board.position, i)
                newPi1 = np.rot90(pi_board1, k=i, axes=(0, 1))
                newPi2 = np.rot90(pi_board2, k=i, axes=(0, 1))
                newPi2 = np.rot90(newPi2,    k=i, axes=(2, 3))
                if j:
                    newB = np.fliplr(newB)
                    newPi1 = np.fliplr(newPi1)
                    newPi2 = np.flip(newPi2, 1)
                    newPi2 = np.flip(newPi2, 3)
                new_board = board
                new_board.position = newB
                new_board.last_move = None
                l += [(new_board, list(newPi1.ravel()) + list(newPi2.ravel()) + [pi_skip])]
        return l

    def stringRepresentation(self, board: Board) -> bytes:
        bn = BoardNumpyUtil.board_to_numpy(board)
        bn[self.n][self.n-1] = 0
        # print(bn)
        return bn.tostring()

    def encode_multiple(self, boards: [Board]) -> np.ndarray:
        """
        Encodes and returns multiple boards
        :param boards: array of boards to encode
        :return: new boards, encoded using onehot encoder
        """
        new_boards = []
        for board in boards:
            new_boards.append(self.encode(board))
        return np.asarray(new_boards)

    def encode(self, board: Board) -> np.ndarray:
        """
        Encode single board for NeuralNet
        :param board: normal board
        :return: new encoded board
        """

        bn = BoardNumpyUtil.board_to_numpy(board)
        bn[self.n][self.n-1] = 0
        return bn

from Game import Game
from .KamisadoLogic import Board, set_grid
import numpy as np
import copy


class KamisadoGame(Game):
    """
    Implement Kamisado (https://boardgamegeek.com/boardgame/38545/kamisado)
    First Phase - Single Round, Standard Rules
    """

    def __init__(self, board_size: int):
        super().__init__()
        self.n = board_size
        set_grid(board_size)  # this is not a pretty way to write this, but I'm lazy

    def getInitBoard(self) -> Board:
        # return initial board
        b = Board()
        return b

    def getBoardSize(self):
        return Board.encode_board_shape()

    def getActionSize(self) -> int:
        # return number of actions
        # Move any piece in the board to any location, skip is included as an action of no movement
        # since each piece is considered unique, the size is the number of pieces for each player*the board size
        return self.n*self.n*3

    def getNextState(self, board: Board, player: int, action: int) -> (Board, int):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        assert action <= self.getActionSize()-1
        # skip
        move = Board.decode_move(action)
        board = copy.copy(board)
        board.execute_move(move, player)
        return board, -player

    def getValidMoves(self, board: Board, player: int) -> np.ndarray:
        # return a fixed size binary vector
        valids = np.zeros(self.getActionSize(), dtype=np.int8)
        legal_moves = board.get_legal_moves(player)
        for m in legal_moves:
            valids[Board.encode_move(m)] = 1
        return valids

    def getGameEnded(self, board: Board, player: int) -> int or float:
        won_color = 0
        # look at player position 0 (black) pieces, and if any of them is on top row they won
        if (board.pieces[0, :, 0] == self.n - 1).any():
            won_color = -1
        # look at player position 1 (white) pieces, and if any of them is on bottom row they won
        elif (board.pieces[1, :, 0] == 0).any():
            won_color = 1
        #print("player {} won".format(won_color))
        if won_color == player:
            return +1
        elif won_color != 0:
            return -1

        if board.has_legal_moves(player):
            return 0  # not over

        # no moves, last player (which is not current player) moved the last legal move (which is an empty move)
        # Therefore last player wins, this player loses
        return -1

    def getCanonicalForm(self, board: Board, player: int) -> Board:
        return board.get_canonical_board(player)

    def getSymmetries(self, board: Board, pi: np.ndarray) -> list:
        # no symmetries
        assert(len(pi) == self.getActionSize())  # -1 for pass
        return [(board, pi)]

    def stringRepresentation(self, board: Board) -> bytes:
        return board.to_string_representation()

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
        return board.encode_board()

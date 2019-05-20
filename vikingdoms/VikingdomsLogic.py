import math
import numpy as np

class Board():
    """
    Author: ranamit112
    Date: May 2rd, 2019.
    Board class.
    Board data:
      first dim is column , 2nd is row:
         pieces[1][3] is the square in column 2,
         at the opposite end of the board in row 3.
      The data for each position is an int of up to 6 bits.
      Highest bit set indicates the number of player pieces in the position (up to 6 (WIN_TOWER))
      Bits after that are the pieces colors - 0 black, 1 white.
      if not bit are set (0) the position is empty.
      Example: b1110 is 3 pieces tall, white white black (from the top)
               b10110 is 4 pieces tall, black white white black (from the top)
    Out-of-board data - each player starts with 12 pieces
    Squares are stored and manipulated as (x,y) tuples.
    x is the column, y is the row. (-1,-1) is considered out-of-board
    Move data - tuple of (number_of_pieces, from_position, to_position)
    """
    WIN_TOWER = 6

    def __init__(self, n, position=None, pieces_left=None, last_move=None):
        """Set up initial board configuration."""

        self.n = n
        self.start_pieces = n * ((n//2)+1)  # 12 for 4*4, 15 for 5*5
        if position is None:
            # Create the empty board array.
            self.position = np.zeros(shape=(n, n), dtype=np.int8)
            self.pieces_left = [self.start_pieces] * 2
            self.last_move = None
        else:
            # copy params
            self.position = position
            self.pieces_left = pieces_left
            self.last_move = last_move

    def __copy__(self):
        return Board(self.n, np.copy(self.position), self.pieces_left.copy(), self.last_move)

    def get_height_of_tower(self, x: int, y: int) -> int:
        v = self.position[x][y]
        return self.get_height_of_tower_val(v)

    @staticmethod
    def get_height_of_tower_val(v: int) -> int:
        h = 0
        while v != 0 and v != 1:
            v = v >> 1
            h += 1
        return h

    def get_top_piece_player(self, x: int, y: int) -> int:
        v = int(self.position[x][y])
        return self.get_top_piece_player_val(v)

    @staticmethod
    def get_top_piece_player_val(v: int) -> int:
        if v == 0:
            return 0
        h = Board.get_height_of_tower_val(v)
        return 1 if ((v >> (h - 1)) & 1) == 1 else -1

    def _is_free_straight(self, x: int, y: int, x2: int, y2: int) -> bool:
        if x > x2:
            for x0 in range(x - 1, x2, -1):
                if self.position[x0][y] != 0:
                    return False
        elif x < x2:
            for x0 in range(x + 1, x2, +1):
                if self.position[x0][y] != 0:
                    return False
        elif y > y2:
            for y0 in range(y - 1, y2, -1):
                if self.position[x][y0] != 0:
                    return False
        elif y < y2:
            for y0 in range(y + 1, y2, +1):
                if self.position[x][y0] != 0:
                    return False
        return True

    def _is_legal_move_straight(self, x: int, y: int, x2: int, y2: int) -> bool:
        if not (((y == y2) and (x != x2)) or
                ((x == x2) and (y != y2))):
            return False

        # xdp = (x2 - x) // abs(x2 - x)
        # ydp = (y2 - y) // abs(y2 - y)
        xdp = 1 if x2 > x else -1
        ydp = 1 if y2 > y else -1
        if xdp != 0:
            for x0 in range(x + xdp, x2, xdp):
                if self.position[x0][y] != 0:
                    return False
        elif ydp != 0:
            for y0 in range(y + ydp, y2, ydp):
                if self.position[x][y0] != 0:
                    return False
        return True

    def _is_legal_move_diag(self, x: int, y: int, x2: int, y2: int) -> bool:
        xd = x - x2
        yd = y - y2
        if xd == 0 or abs(xd) != abs(yd):
            return False
        xdp = 1 if x2 > x else -1
        ydp = 1 if y2 > y else -1

        for p in range(1, abs(xd)):
            if self.position[x + xdp*p][y + ydp*p] != 0:
                return False
        return True

    def _is_legal_move_knight(self, x: int, y: int, x2: int, y2: int) -> bool:
        xd = x - x2
        yd = y - y2
        if (
                (abs(xd) == 0 or abs(yd) == 0) or
                (abs(xd) >= 3 or abs(yd) >= 3) or
                (abs(xd) == abs(yd))):
            return False
        return True

    def _is_legal_piece_move(self, x: int, y: int, x2: int, y2: int, h: int) -> bool:
        """
        Implements the move-part of tower legality check.
        Move pieces from a tower of height h from x,y to x2,y2
        Assumes all position are valid, and the board has these pieces
        Does not check max number of pieces in position
        """
        if h == 1:
            return (
                    ((y == y2) and (x == (x2 + 1) or x == (x2 - 1))) or
                    ((x == x2) and (y == (y2 + 1) or y == (y2 - 1)))
                    )
        elif h == 2:
            return self._is_legal_move_straight(x, y, x2, y2)
        elif h == 3:
            return self._is_legal_move_knight(x, y, x2, y2)
        elif h == 4:
            return self._is_legal_move_diag(x, y, x2, y2)
        elif h == 5:
            return self._is_legal_move_straight(x, y, x2, y2) or self._is_legal_move_diag(x, y, x2, y2)
        return False

    def get_legal_moves(self, color: int) -> list:
        """
        Returns all the legal moves for the given color.
        moves are in the form of (int, (int, int), (int, int))
        """
        moves = list()  # stores the legal moves.
        # move piece from out-of-board to each free position
        color_pos = 1 if color == 1 else 0
        if self.pieces_left[color_pos] > 0:
            for y in range(self.n):
                for x in range(self.n):
                    if self.position[x][y] == 0:
                        moves.append((1, (-1, -1), (x, y)))

        # move part of tower from one place on the board to another (which is not empty)
        for y in range(self.n):
            for x in range(self.n):
                h = self.get_height_of_tower(x, y)
                if h == 0:
                    continue
                for y2 in range(self.n):
                    for x2 in range(self.n):
                        if x == x2 and y == y2:
                            continue
                        if self.last_move[2] == (x,y) and self.last_move[1] == (x2,y2):
                            # print("trying undo-last-move {},{}->{},{}".format(x,y,x2,y2))
                            continue
                        if self.position[x2][y2] == 0:
                            continue
                        h2 = self.get_height_of_tower(x2, y2)
                        if h2 >= Board.WIN_TOWER:
                            assert h2 < Board.WIN_TOWER, "Moving to position {},{} with height={}".format(x2, y2, h2)
                            continue
                        if not self._is_legal_piece_move(x, y, x2, y2, h):
                            continue
                        for h0 in range(1, min(h+1, Board.WIN_TOWER-h2+1)):
                            moves.append((h0, (x, y), (x2, y2)))

        return moves

    def has_legal_moves(self) -> bool:
        """
        Returns True if has legal move else False
        """
        # I think there is always a legal move for one of the colors
        return True

    def execute_move(self, move: (int, (int, int), (int, int)), color: int) -> None:
        """Perform the given move on the board; flips pieces as necessary.
        color gives the color pf the piece to play (1=white,-1=black)
        """

        color_pos = 1 if color == 1 else 0
        move_height, (x, y), (x2, y2) = move
        if x == -1 and y == -1:
            assert move_height == 1
            assert self.position[x2][y2] == 0, "{},{} is not empty".format(x2, y2)
            assert self.pieces_left[color_pos] >= 1 , "color pieces {} == {}".format(color_pos,self.pieces_left[color_pos])
            self.position[x2][y2] = color_pos | (1 << 1)
            self.pieces_left[color_pos] -= 1
        else:
            v = self.position[x][y]
            v2 = self.position[x2][y2]
            h = self.get_height_of_tower(x, y)
            assert 1 <= move_height <= h
            if move_height == h and v2 == 0:
                # just move the value
                v2 = v
                v = 0
            else:
                # take v2, remove the height bit,
                # take the moved bits from top of v to v2, reposition for new height
                # add new top bit
                h2 = self.get_height_of_tower(x2, y2)
                assert move_height + h2 <= self.WIN_TOWER
                v2 = v2 & ~(1 << h2) | \
                    (((((1 << move_height) - 1) << (h - move_height) & v) >> h) << h2) | \
                    (1 << (h2 + move_height))
                if move_height == h:
                    v = 0
                else:
                    # take v, keep only bottom bits (removes the height bit)
                    # add new top bit
                    v = (((1 << (h - move_height)) - 1) & v) | (1 << (h - move_height))
            self.position[x][y] = v
            self.position[x2][y2] = v2
        self.last_move = move

class BoardNumpyUtil():
    @staticmethod
    def board_shape_numpy(n: int):
        return (n+1, n)

    @staticmethod
    def board_to_numpy(board: Board) -> np.ndarray:
        b2 = np.zeros(shape=(1, board.n), dtype=np.int8)
        b2[0][0:2] = board.pieces_left[:]
        last_move_val = BoardNumpyUtil.move_to_numpy_val(board.n,
                                                         board.last_move) if board.last_move is not None else -1
        b2[0][board.n-1] = last_move_val
        return np.concatenate([board.position, b2])

    @staticmethod
    def board_from_numpy(board: np.ndarray) -> Board:
        #print(board)
        n = board.shape[0]-1
        b = Board(n)
        b.position = board[0:n].copy()
        b.pieces_left[0:2] = board[n][0:2].tolist()
        mv = board[n][n-1]
        b.last_move = BoardNumpyUtil.move_from_numpy_val(n, mv) if mv != -1 else None
        return b

    @staticmethod
    def move_to_numpy_val(n: int, action: (int, (int, int), (int, int))) -> int:
        piece_count, from_pos, to_pos = action
        if from_pos == (-1, -1):
            x2, y2 = to_pos
            return x2 + (y2 * n)
        else:
            x, y = from_pos
            x2, y2 = to_pos
            return n**2 + ((x + (y * n)) + (x2 + (y2 * n))*(n**2))+((piece_count-1)*(n**4))

    @staticmethod
    def move_from_numpy_val(n: int, val: int) -> (int, (int, int), (int, int)):
        if 0 <= val < n**2:
            return 1, (-1, -1), (val % n, val // n)
        else:
            v2 = val - n**2
            pv = v2 % (n**2)
            p2v = (v2 // (n**2)) % (n**2)
            return (v2 // (n**4))+1, (pv % n, pv // n), (p2v % n, p2v // n)

    @staticmethod
    def get_height_of_tower_val(v: int) -> int:
        return Board.get_height_of_tower_val(v)

    @staticmethod
    def _get_reverse_val(v: int) -> int:
        if v == 0:
            return 0
        h = BoardNumpyUtil.get_height_of_tower_val(v)
        # set height bit, reverse all other valid bits
        return (1 << h) | (((1 << h) - 1) & ~ (v & 0x1f))

    @staticmethod
    def get_canonical_board(n: int, board: np.ndarray, player: int) -> np.ndarray:
        # return state if player==1, else return -state if player==-1
        if player == 1:
            return board

        r = np.copy(board)
        vhf = np.vectorize(BoardNumpyUtil._get_reverse_val)
        # get reverse values of position, switch the pieces-left position
        r[0:n] = vhf(board[0:n])
        r[n][0], r[n][1] = r[n][1], r[n][0]
        return r


def display(board):
    # TODO: impl
    n = board.shape[0]

    for y in range(n):
        print(y, "|", end="")
    print("")
    print(" -----------------------")
    for y in range(n):
        print(y, "|", end="")    # print the row #
        for x in range(n):
            piece = board[y][x]    # get the piece to print
            if piece == -1:
                print("b ", end="")
            elif piece == 1:
                print("W ", end="")
            else:
                if x == n:
                    print("-", end="")
                else:
                    print("- ", end="")
        print("|")

    print("   -----------------------")

import numpy as np
import copy
import typing
from enum import Enum


#class Colors(Enum):
NONE = -1


class Grid8:
    N = 8
    ORANGE = 0
    BLUE = 1
    PURPLE = 2
    PINK = 3
    YELLOW = 4
    RED = 5
    GREEN = 6
    BROWN = 7
    NUM_COLORS = 8
    colors = np.array(
        [[ORANGE, BLUE  , PURPLE, PINK  , YELLOW, RED   , GREEN , BROWN ],
         [RED   , ORANGE, PINK  , GREEN , BLUE  , YELLOW, BROWN , PURPLE],
         [GREEN , PINK  , ORANGE, RED   , PURPLE, BROWN , YELLOW, BLUE  ],
         [PINK  , PURPLE, BLUE  , ORANGE, BROWN , GREEN , RED   , YELLOW],
         [YELLOW, RED   , GREEN , BROWN , ORANGE, BLUE  , PURPLE, PINK  ],
         [BLUE  , YELLOW, BROWN , PURPLE, RED   , ORANGE, PINK  , GREEN ],
         [PURPLE, BROWN , YELLOW, BLUE  , GREEN , PINK  , ORANGE, RED   ],
         [BROWN , GREEN , RED   , YELLOW, PINK  , PURPLE, BLUE  , ORANGE]])


class Grid6:
    N = 6
    ORANGE = 0
    BLUE = 1
    GREEN = 2
    PURPLE = 3
    YELLOW = 4
    RED = 5
    NUM_COLORS = 6
    colors = np.array(
        [[ORANGE, BLUE  , GREEN , PURPLE, YELLOW, RED   ],
         [PURPLE, ORANGE, YELLOW, BLUE  , RED   , GREEN ],
         [YELLOW, GREEN , ORANGE, RED   , PURPLE, BLUE  ],
         [BLUE  , PURPLE, RED   , ORANGE, GREEN , YELLOW],
         [GREEN , RED   , BLUE  , YELLOW, ORANGE, PURPLE],
         [RED   , YELLOW, PURPLE, GREEN , BLUE  , ORANGE],
         ])


Grid: typing.Union[Grid6, Grid8, None] = None


def set_grid(n: int):
    assert n == 8 or n == 6
    global Grid
    if Grid is None:
        Grid = Grid8 if n == 8 else Grid6
    else:
        raise RuntimeError("Double call to set_grid")


class Board:
    """
    Author: ranamit112
    Date: Oct 21th, 2019.
    Board class.
    Board data:
      first dim is row ,2nd is column:
         pieces[1][3] is the square in the second row, forth position from the left
    """

    def __init__(self, pieces=None, last_move=None, last_color=NONE):
        """Set up initial board configuration."""
        if pieces is None:
            # Create the empty board array.
            # white is 1, black is -1 (or 0). White is bottom, going up. Black is on top, going down
            #self.pieces = np.array(
            #    [[[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7]],
            #     [[7, 7], [7, 6], [7, 5], [7, 4], [7, 3], [7, 2], [7, 1], [7, 0]]], dtype=np.int8)
            self.pieces = np.array(
                 [[[0, i] for i in range(Grid.N)],
                  [[Grid.N - 1, i] for i in reversed(range(Grid.N))]], dtype=np.int8)
            self.last_piece_color = NONE
            self.last_move = None
        else:
            # copy params
            self.pieces = pieces
            self.last_piece_color = last_color
            self.last_move = last_move

    def __copy__(self):
        return Board(self.pieces.copy(), copy.copy(self.last_move), self.last_piece_color)

    MoveType = typing.Tuple[int, int, int]
    MoveList = typing.List[MoveType]

    def get_legal_moves(self, player: int) -> MoveList:
        """
        Returns all the legal moves for the given player color.
        Moves are in the form of (int, ( int, int))),
        where the first value is the Color of the piece, then (y2, x2)
        """
        # move piece from out-of-board to each free position
        #color_pos = 1 if color == 1 else 0
        if self.last_piece_color == NONE:  # starting position
            moves = list()  # stores the legal moves, from all pieces
            for c in range(Grid.NUM_COLORS):
                moves.extend(self.get_moves_for_piece(player, c))
            return moves
        return self.get_moves_for_piece(player, self.last_piece_color)

    def get_moves_for_piece(self, player, last_piece_color) -> MoveList:
        player_pos = 1 if player == 1 else 0
        moves = list()  # stores the legal moves.
        selected_piece = [player_pos, last_piece_color]
        selected_piece_pos = self.pieces[player_pos, last_piece_color]
        y, x = selected_piece_pos
        pos = (y, x)
        # black (player!=0) moves down (1 by row index) and white (player=1) moves up (-1 by row index)
        y_dir = -1 if player == 1 else 1
        for x_dir in [-1, 0, 1]:
            for i in range(1, Grid.N):
                x2 = x + (i * x_dir * y_dir)
                y2 = y + (i * y_dir)
                # out of bounds
                if not ((0 <= x2 < Grid.N) and (0 <= y2 < Grid.N)):
                    break
                # hit another piece - this searches for a match in the pieces locations 3d array
                pos2 = (y2, x2)
                if (self.pieces == pos2).all(axis=2).any():
                    break
                moves.append((last_piece_color, x_dir, i))
        if len(moves) == 0:
            # the empty move is added
            moves.append((last_piece_color, 0, 0))
        return moves

    # noinspection PyMethodMayBeStatic
    def has_legal_moves(self, player: int) -> bool:
        """
        Returns True if has legal move else False
        """
        if self.last_move and self.last_move[2] == 0:  # empty move
            b_moves = self.get_moves_for_piece(player, self.last_piece_color)
            w_moves = self.get_moves_for_piece(-player, self.last_move[0])
            # b_moves = len(self.get_legal_moves(1))
            # w_moves = len(self.get_legal_moves(-1))
            if w_moves != 1 or w_moves[0][2] != 0:
                pass
            if len(b_moves) == 1 and b_moves[0][2] == 0 and len(w_moves) == 1 and w_moves[0][2] == 0:
                return False
        return True

    def execute_move(self, move: MoveType, player: int) -> None:
        """Perform the given move on the board;
        player the player to play (1=white,-1=black)
        Moves are in the form of (int, (int, int)), where the first value is the Color of the piece, then (y, x)
        """
        color_pos = 1 if player == 1 else 0
        y_dir = 1 if color_pos == 0 else -1
        piece_color, x_dir, y_diff = move
        y, x = self.pieces[color_pos][piece_color]
        y2 = y + y_diff*y_dir
        x2 = x + y_diff*x_dir*y_dir
        pos = [y, x]
        pos2 = [y2, x2]
        assert pos == pos2 or not (self.pieces == pos2).all(axis=2).any()
        self.pieces[color_pos][piece_color] = pos2
        self.last_piece_color = Grid.colors[y2, x2]
        self.last_move = move

    def get_canonical_board(self, player: int) -> 'Board':
        # return state if player==1, else return -state if player==-1
        b = copy.copy(self)
        if player == 1:
            return b
        # as each row is a different player's set of positions, flipping the rows switches all player pieces
        b.pieces = np.flipud(b.pieces)
        # rotate 180 degrees all positions, by "flipping" every x and y value
        # the "flip" is performed by subtracting current value from the max value of grid (8) - 1. So 0->7, 1->6 etc.
        b.pieces = (Grid.N - 1) - b.pieces
        return b

    def to_string_representation(self) -> bytes:
        # convert to packed binary representation
        return self.pieces.tostring()+np.array([self.last_piece_color], dtype=np.int8).tostring()

    def debug_display(self):
        a = np.zeros(shape=(Grid.N, Grid.N), dtype=np.int8)
        #a[self.pieces[0][:, 0], self.pieces[0][:, 1]] = 1
        #a[self.pieces[1][:, 0], self.pieces[1][:, 1]] = -1
        for i, loc in enumerate(self.pieces[0]):
            a[loc[0], loc[1]] = i + 1
        for i, loc in enumerate(self.pieces[1]):
            a[loc[0], loc[1]] = -1*(i + 1)
        print(a)
        print("last played {}".format(self.last_piece_color))

    def excel_display(self):
        a = np.zeros(shape=(Grid.N, Grid.N), dtype=np.int8)
        #a[self.pieces[0][:, 0], self.pieces[0][:, 1]] = 1
        #a[self.pieces[1][:, 0], self.pieces[1][:, 1]] = -1
        for i, loc in enumerate(self.pieces[0]):
            a[loc[0], loc[1]] = i + 1
        for i, loc in enumerate(self.pieces[1]):
            a[loc[0], loc[1]] = -1*(i + 1)

        def to_excel_value(v):
            if v == 0:
                return ""
            return format(("B{}" if v > 0 else "W{}").format(abs(v)))

        b = np.vectorize(to_excel_value)(a)
        s = str(b)
        s += "\nlast played {}".format(self.last_piece_color)
        s = s.replace("'", "").replace("]", "")
        return s

    @staticmethod
    def encode_board_shape():
        return Grid.N, Grid.N, Grid.N*2 + 1

    def encode_board(self) -> np.ndarray:
        b = np.zeros(shape=self.encode_board_shape(), dtype=np.int8)
        # encode the number of pieces left, each as a whole layer of same-value, between 0 and 1
        for i, loc in enumerate(self.pieces[0]):
            b[loc[0], loc[1], i] = 1
        for i, loc in enumerate(self.pieces[1]):
            b[loc[0], loc[1], i+Grid.N] = -1
        if self.last_piece_color == NONE:
            b[:, :, Grid.N * 2] = 1
        else:
            b[:, :, Grid.N * 2][Grid.colors == self.last_piece_color] = 1

        return b

    @staticmethod
    def encode_move(move: MoveType) -> int:
        piece_color, x_dir, y_diff = move
        return ((Grid.N**2)*(x_dir + 1)) + (Grid.N*y_diff) + piece_color

    @staticmethod
    def decode_move(val: int) -> MoveType:
        x_dir = (val // (Grid.N ** 2)) - 1
        y_diff = (val // Grid.N) % Grid.N
        piece_color = val % Grid.N
        move = piece_color, x_dir, y_diff
        return move

    def display(self):
        raise NotImplementedError()

        def piece_display(piece: np.array):
            PRINT_MAP = {0: ' ', 1: '●', -1: '○'}
            return PRINT_MAP.get(piece, ' ')

        import beautifultable
        table = beautifultable.BeautifulTable()

        for y in range(self.n):
            table.insert_row(y, ['\n'.join(reversed(list(map(piece_display, self.position[x, y])))) for x in range(self.n)])

        table.column_headers = DisplayRowNotation.get_row_names(self.n)
        column_names = DisplayRowNotation.get_column_names(self.n)
        if column_names:
            table.insert_column(0, '', column_names)
        table.set_style(beautifultable.STYLE_BOX)
        print(table)
        print("")
        print("{} x ●, {} x ○".format(self.pieces_left[1], self.pieces_left[0]))


class DisplayRowNotation:
    NO_NOTATION = 0
    EXCEL_NOTATION = 1
    CHESS_NOTATION = 2

    @staticmethod
    def get_row_names(n):
        if DISPLAY_NOTATION in [DisplayRowNotation.EXCEL_NOTATION, DisplayRowNotation.CHESS_NOTATION]:
            return [chr(ord('A') + i) for i in range(n)]
        #elif DISPLAY_NOTATION == DisplayRowNotation.CHESS_NOTATION:
        #    return [i + 1 for i in range(n)]
        return None

    @staticmethod
    def get_column_names(n):
        if DISPLAY_NOTATION == DisplayRowNotation.EXCEL_NOTATION:
            return [i + 1 for i in range(n)]
        elif DISPLAY_NOTATION == DisplayRowNotation.CHESS_NOTATION:
            return [n - i for i in range(n)]
        return None

    @staticmethod
    def get_position(n, x, y):
        if DISPLAY_NOTATION == DisplayRowNotation.EXCEL_NOTATION:
            return "{}{}".format(chr(ord('A') + x), y+1)
        elif DISPLAY_NOTATION == DisplayRowNotation.CHESS_NOTATION:
            return "{}{}".format(chr(ord('A') + x), n-y)
        return "{},{}".format(x, y)


DISPLAY_NOTATION = DisplayRowNotation.EXCEL_NOTATION

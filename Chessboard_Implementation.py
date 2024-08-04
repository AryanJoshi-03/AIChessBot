import pygame
import sys
import copy

# Initialize Pygame
pygame.init()

# Define the dimensions of the game window
WIDTH = 800
HEIGHT = 800
size = WIDTH, HEIGHT

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
LIGHT_BROWN = (222, 184, 135)
DARK_BROWN = (139, 69, 19)
HIGHLIGHT_COLOR = (0, 255, 0)

# Create the game window
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Chess")

# Load chess pieces (You need to update these paths based on your system)
pieces = {
    "white_pawn": pygame.image.load("/Users/aryanjoshi/Downloads/pieces-png/white-pawn.png"),
    "black_pawn": pygame.image.load("/Users/aryanjoshi/Downloads/pieces-png/black-pawn.png"),
    "white_king": pygame.image.load("/Users/aryanjoshi/Downloads/pieces-png/white-king.png"),
    "black_king": pygame.image.load("/Users/aryanjoshi/Downloads/pieces-png/black-king.png"),
    "white_queen": pygame.image.load("/Users/aryanjoshi/Downloads/pieces-png/white-queen.png"),
    "black_queen": pygame.image.load("/Users/aryanjoshi/Downloads/pieces-png/black-queen.png"),
    "white_bishop": pygame.image.load("/Users/aryanjoshi/Downloads/pieces-png/white-bishop.png"),
    "black_bishop": pygame.image.load("/Users/aryanjoshi/Downloads/pieces-png/black-bishop.png"),
    "white_knight": pygame.image.load("/Users/aryanjoshi/Downloads/pieces-png/white-knight.png"),
    "black_knight": pygame.image.load("/Users/aryanjoshi/Downloads/pieces-png/black-knight.png"),
    "white_rook": pygame.image.load("/Users/aryanjoshi/Downloads/pieces-png/white-rook.png"),
    "black_rook": pygame.image.load("/Users/aryanjoshi/Downloads/pieces-png/black-rook.png"),
}

# Resizing the images to fit in the square_size
square_size = 100
for key in pieces:
    pieces[key] = pygame.transform.scale(pieces[key], (square_size, square_size))

def draw_board(screen):
    for row in range(8):
        for col in range(8):
            square_x = col * square_size
            square_y = row * square_size
            square_color = LIGHT_BROWN if (row + col) % 2 == 0 else DARK_BROWN
            pygame.draw.rect(screen, square_color, (square_x, square_y, square_size, square_size))

def draw_pieces(screen, board):
    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            if piece:
                piece_key = f"{piece.color}_{piece.piece_type}"
                screen.blit(pieces[piece_key], (col * square_size, row * square_size))

def draw_highlight(screen, row, col):
    pygame.draw.rect(screen, HIGHLIGHT_COLOR, (col * square_size, row * square_size, square_size, square_size), 5)

def get_square_under_mouse():
    mouse_pos = pygame.Vector2(pygame.mouse.get_pos())
    x, y = [int(v // square_size) for v in mouse_pos]
    return x, y

# Utility and Piece Movement Functions

def is_within_boundaries(row,col):
    return 0 <= row < 8 and 0 <= col < 8

def is_valid_square(board, row, col, piece_color):
    return 0 <= row < 8 and 0 <= col < 8 and (board[row][col] is None or board[row][col].color != piece_color)

def pawn_moves(board, piece):
    valid_moves = []
    row, col = piece.position
    direction = -1 if piece.color == 'white' else 1
    start_row = 6 if piece.color == 'white' else 1

    # Move forward
    if is_valid_square(board, row + direction, col, is_valid_square) and board[row + direction][col] is None:
        valid_moves.append((row + direction, col))
        # Check if it's at the starting position and can move two squares
        if row == start_row and is_valid_square(board, row + 2 * direction, col, is_valid_square) and board[row + direction][col] is None and board[row + 2*direction][col] is None:
            valid_moves.append((row + 2 * direction, col))

    # Capturing moves
    for offset in [-1, 1]:
        capture_col = col + offset
        if 0 <= capture_col < 8:
            capture_square = board[row + direction][capture_col]
            if capture_square and capture_square.color != piece.color:
                valid_moves.append((row + direction, capture_col))
            elif board[row][capture_col] and board[row][capture_col].piece_type == 'pawn' and board[row][capture_col].color != piece.color:
                if piece.color == 'white' and row == 3 or piece.color == 'black' and row == 4:
                    last_move_start, last_move_end = last_move
                    if last_move_end == (row, capture_col) and abs(last_move_start[0] - last_move_end[0]) == 2:
                        #Game loop accounts for captured pawn
                        valid_moves.append((row + direction, capture_col))

    return valid_moves

def knight_moves(board, piece):
    valid_moves = []
    row, col = piece.position

    # Possible knight moves relative to rows, col
    possible_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (2, -1), (2, 1), (1, 2), (1, -2)]

    # Adjusting coordinates accordingly
    for r, c in possible_moves:
        new_row = row + r
        new_col = col + c

        if is_valid_square(board, new_row, new_col,piece.color):
            valid_moves.append((new_row, new_col))

    return valid_moves

def bishop_moves(board, piece):
    valid_moves = []
    row, col = piece.position
    possible_moves = [(1, 1), (-1, 1), (1, -1), (-1, -1)]

    for r, c in possible_moves:
        new_row = row + r
        new_col = col + c

        while is_valid_square(board, new_row, new_col,piece.color):
            if board[new_row][new_col] is None:
                valid_moves.append((new_row, new_col))

            elif board[new_row][new_col].color != piece.color:
                valid_moves.append((new_row,new_col))
                break

            new_row = new_row + r
            new_col = new_col + c

    return valid_moves

def rook_moves(board, piece):
    valid_moves = []
    row, col = piece.position
    possible_moves = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    for r, c in possible_moves:
        new_row = row + r
        new_col = col + c

        while is_valid_square(board, new_row, new_col,piece.color):
            if board[new_row][new_col] is None:
                valid_moves.append((new_row, new_col))
                
            elif board[new_row][new_col].color != piece.color:
                valid_moves.append((new_row,new_col))
                break

            new_row = new_row + r
            new_col = new_col + c

    return valid_moves

def queen_moves(board, piece):
    valid_moves = []
    row, col = piece.position
    possible_moves = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]

    for r, c in possible_moves:
        new_row = row + r
        new_col = col + c

        while is_valid_square(board, new_row, new_col,piece.color):
            if board[new_row][new_col] is None:
                valid_moves.append((new_row, new_col))
                
            elif board[new_row][new_col].color != piece.color:
                valid_moves.append((new_row,new_col))
                break

            new_row = new_row + r
            new_col = new_col + c

    return valid_moves

def king_moves(board, piece):
    valid_moves = []
    row, col = piece.position
    possible_moves = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]

    for r, c in possible_moves:
        new_row = row + r
        new_col = col + c

        if is_valid_square(board, new_row, new_col,piece.color):
            valid_moves.append((new_row, new_col))

    return valid_moves

def is_in_check(board, color):
    king_position = None
    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            if piece and piece.color == color and piece.piece_type == 'king':
                king_position = (row, col)
                break
        if king_position:
            break

    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            if piece and piece.color != color:
                valid_moves = piece.get_moves(board)
                if king_position in valid_moves:
                    return True
    return False
def is_checkmate(board, color):
    if not is_in_check(board, color):
        return False

    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            if piece and piece.color == color:
                valid_moves = piece.get_moves(board)
                for move in valid_moves:
                    temp_board = copy.deepcopy(board)
                    temp_piece = temp_board[row][col]
                    temp_piece.update_position(move)
                    temp_board[move[0]][move[1]] = temp_piece
                    temp_board[row][col] = None
                    if not is_in_check(temp_board, color):
                        return False
    return True

def is_stalemate(board, color):
    if is_in_check(board, color):
        return False

    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            if piece and piece.color == color:
                valid_moves = piece.get_moves(board)
                for move in valid_moves:
                    temp_board = copy.deepcopy(board)
                    temp_piece = temp_board[row][col]
                    temp_piece.update_position(move)
                    temp_board[move[0]][move[1]] = temp_piece
                    temp_board[row][col] = None
                    if not is_in_check(temp_board, color):
                        return False
    return True

# Utility function to check if a move is legal
def is_legal_move(board, piece, target_position):
    original_position = piece.position
    temp_board = copy.deepcopy(board)
    temp_piece = temp_board[original_position[0]][original_position[1]]
    temp_piece.update_position(target_position)
    temp_board[target_position[0]][target_position[1]] = temp_piece
    temp_board[original_position[0]][original_position[1]] = None
    return not is_in_check(temp_board, piece.color)
def can_castle(board, king, rook):
    if king.has_moved or rook.has_moved:
        return False
    if king.color != rook.color:
        return False
    y = king.position[0]
    king_x = king.position[1]
    rook_x = rook.position[1]

    # Check if squares between king and rook are empty
    if rook_x == 0:  # Queen-side castling
        for x in range(1, king_x):
            if board[y][x] is not None:
                return False
    elif rook_x == 7:  # King-side castling
        for x in range(king_x + 1, 7):
            if board[y][x] is not None:
                return False

    # Check if squares the king passes through are under attack
    if rook_x == 0:  # Queen-side castling
        squares_to_check = [(y, king_x), (y, king_x - 1), (y, king_x - 2)]
    elif rook_x == 7:  # King-side castling
        squares_to_check = [(y, king_x), (y, king_x + 1), (y, king_x + 2)]

    for square in squares_to_check:
        if is_square_under_attack(board, square, king.color):
            return False

    return True

def is_square_under_attack(board, square, color):
    enemy_color = 'black' if color == 'white' else 'white'
    for row in board:
        for piece in row:
            if piece and piece.color == enemy_color:
                if square in get_valid_moves(piece):
                    return True
    return False


def get_castling_moves(board, king):
    castling_moves = []
    y = king.position[0]
    if king.color == 'white':
        if can_castle(board, king, board[y][0]):
            castling_moves.append((y, 2))  # Queen-side
        if can_castle(board, king, board[y][7]):
            castling_moves.append((y, 6))  # King-side
    elif king.color == 'black':
        if can_castle(board, king, board[y][0]):
            castling_moves.append((y, 2))  # Queen-side
        if can_castle(board, king, board[y][7]):
            castling_moves.append((y, 6))  # King-side
    return castling_moves

def evaluate_board(board):
    piece_value = {
        'pawn': 1,
        'knight': 3,
        'bishop': 3,
        'rook': 5,
        'queen': 9,
        'king': 0  # King's value is usually not considered in simple evaluation
    }
    
    score = 0
    for row in board:
        for piece in row:
            if piece is not None:
                if piece.color == 'white':
                    score += piece_value[piece.piece_type]
                else:
                    score -= piece_value[piece.piece_type]
    return score

def minimax(board, depth, alpha, beta, maximizing_player):
    if depth == 0 or is_checkmate(board, 'white') or is_checkmate(board, 'black'):
        return evaluate_board(board), None

    if maximizing_player:
        max_eval = float('-inf')
        best_move = None
        for move in get_all_valid_moves(board, 'white'):
            new_board = make_move(board, move)
            eval, _ = minimax(new_board, depth - 1, alpha, beta, False)
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        best_move = None
        for move in get_all_valid_moves(board, 'black'):
            new_board = make_move(board, move)
            eval, _ = minimax(new_board, depth - 1, alpha, beta, True)
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move




class Piece:
    def __init__(self, piece_type, color, position):
        self.piece_type = piece_type
        self.color = color
        self.position = position
        self.has_moved = False

    def get_moves(self, board):
        if self.piece_type == 'pawn':
            return pawn_moves(board, self)
        elif self.piece_type == 'knight':
            return knight_moves(board, self)
        elif self.piece_type == 'bishop':
            return bishop_moves(board, self)
        elif self.piece_type == 'rook':
            return rook_moves(board, self)
        elif self.piece_type == 'queen':
            return queen_moves(board, self)
        elif self.piece_type == 'king':
            return king_moves(board, self)
        else:
            return []
    
    def update_position(self, new_position):
        self.position = new_position
        self.has_moved = True

class Queen(Piece):
    def __init__(self, color, position):
        super().__init__('queen', color, position)

class Rook(Piece):
    def __init__(self, color, position):
        super().__init__('rook', color, position)

class Bishop(Piece):
    def __init__(self, color, position):
        super().__init__('bishop', color, position)

class Knight(Piece):
    def __init__(self, color, position):
        super().__init__('knight', color, position)


# Initial board setup
board = [
    [Piece('rook', 'black', (0, 0)), Piece('knight', 'black', (0, 1)), Piece('bishop', 'black', (0, 2)), Piece('queen', 'black', (0, 3)), Piece('king', 'black', (0, 4)), Piece('bishop', 'black', (0, 5)), Piece('knight', 'black', (0, 6)), Piece('rook', 'black', (0, 7))],
    [Piece('pawn', 'black', (1, i)) for i in range(8)],
    [None] * 8,
    [None] * 8,
    [None] * 8,
    [None] * 8,
    [Piece('pawn', 'white', (6, i)) for i in range(8)],
    [Piece('rook', 'white', (7, 0)), Piece('knight', 'white', (7, 1)), Piece('bishop', 'white', (7, 2)), Piece('queen', 'white', (7, 3)), Piece('king', 'white', (7, 4)), Piece('bishop', 'white', (7, 5)), Piece('knight', 'white', (7, 6)), Piece('rook', 'white', (7, 7))]
]

selected_piece = None
highlighted_moves = []
turn = 'white'
last_move = None
promotion_pending = False
promotion_piece = None
promotion_position = None
running = True

def get_valid_moves(piece):
    return piece.get_moves(board)

# Game loop
def promote_pawn(piece, choice):
    position = piece.position
    if choice == 'Q':
        return Queen(piece.color, position)
    elif choice == 'R':
        return Rook(piece.color, position)
    elif choice == 'B':
        return Bishop(piece.color, position)
    elif choice == 'N':
        return Knight(piece.color, position)
    

def get_all_valid_moves(board, color):
    moves = []
    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            if piece and piece.color == color:
                moves.extend([(piece.position, move) for move in piece.get_moves(board)])
    return moves

def make_move(board, move):
    (start_pos, end_pos) = move
    new_board = copy.deepcopy(board)
    piece = new_board[start_pos[0]][start_pos[1]]
    piece.update_position(end_pos)
    new_board[end_pos[0]][end_pos[1]] = piece
    new_board[start_pos[0]][start_pos[1]] = None
    return new_board

def get_best_move(board, depth):
    best_move = None
    best_value = float('-inf')
    for move in get_all_valid_moves(board, 'black'):
        new_board = make_move(board, move)
        board_value = minimax(new_board, depth - 1, float('-inf'), float('inf'), False)
        if board_value > best_value:
            best_value = board_value
            best_move = move
    return best_move
# Game loop

# Game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = get_square_under_mouse()

            if selected_piece:
                if (y, x) in highlighted_moves:
                    piece = board[selected_piece[0]][selected_piece[1]]
                    target_position = (y, x)

                    # Check if the move is legal
                    if is_legal_move(board, piece, target_position):
                        if piece.piece_type == 'pawn' and abs(piece.position[0] - y) == 1 and abs(piece.position[1] - x) == 1 and board[y][x] is None:
                            board[selected_piece[0]][x] = None  # Remove the captured pawn for en passant
                        last_move = (piece.position, target_position)  # Track the last move

                        # Handle castling
                        if piece.piece_type == 'king' and abs(piece.position[1] - x) == 2:
                            if x > piece.position[1]:  # King-side castling
                                rook = board[y][7]
                                if can_castle(board, piece, rook):
                                    board[y][5] = rook
                                    rook.update_position((y, 5))
                                    board[y][7] = None
                            else:  # Queen-side castling
                                rook = board[y][0]
                                if can_castle(board, piece, rook):
                                    board[y][3] = rook
                                    rook.update_position((y, 3))
                                    board[y][0] = None

                        piece.update_position(target_position)
                        board[y][x] = piece
                        board[selected_piece[0]][selected_piece[1]] = None

                        # Handle promotion
                        if piece.piece_type == 'pawn' and (y == 0 or y == 7):
                            promotion_pending = True
                            promotion_piece = piece
                            promotion_position = (y, x)
                        else:
                            turn = 'black' if turn == 'white' else 'white'

                        selected_piece = None
                        highlighted_moves = []

                        # Check for check, checkmate, and stalemate after each move
                        if is_checkmate(board, turn):
                            print(f"Checkmate! {turn} loses.")
                            running = False
                        elif is_stalemate(board, turn):
                            print("Stalemate! It's a draw.")
                            running = False
                        elif is_in_check(board, turn):
                            print(f"{turn} is in check.")
                    else:
                        print("Illegal move.")
                        selected_piece = None
                        highlighted_moves = []

                else:
                    selected_piece = None
                    highlighted_moves = []
            else:
                if board[y][x] and board[y][x].color == turn:
                    selected_piece = (y, x)
                    highlighted_moves = get_valid_moves(board[y][x])
                    if board[y][x].piece_type == 'king':
                        highlighted_moves.extend(get_castling_moves(board, board[y][x]))

        elif event.type == pygame.KEYDOWN:
            if promotion_pending:
                key = pygame.key.name(event.key).upper()
                if key in ['Q', 'R', 'B', 'N']:
                    board[promotion_position[0]][promotion_position[1]] = promote_pawn(promotion_piece, key)
                    promotion_pending = False
                    turn = 'black' if turn == 'white' else 'white'

    # AI move generation for black
    if turn == 'black' and not promotion_pending:
        _,best_move = minimax(board, 3, -float('inf'), float('inf'), False)
        if best_move:
            piece = board[best_move[0][0]][best_move[0][1]]
            target_position = best_move[1]

            # Handle castling
            if piece.piece_type == 'king' and abs(piece.position[1] - target_position[1]) == 2:
                if target_position[1] > piece.position[1]:  # King-side castling
                    rook = board[target_position[0]][7]
                    if can_castle(board, piece, rook):
                        board[target_position[0]][target_position[1] - 1] = rook
                        rook.update_position((target_position[0], target_position[1] - 1))
                        board[target_position[0]][7] = None
                else:  # Queen-side castling
                    rook = board[target_position[0]][0]
                    if can_castle(board, piece, rook):
                        board[target_position[0]][target_position[1] + 1] = rook
                        rook.update_position((target_position[0], target_position[1] + 1))
                        board[target_position[0]][0] = None

            piece.update_position(target_position)
            board[target_position[0]][target_position[1]] = piece
            board[best_move[0][0]][best_move[0][1]] = None

            # Handle promotion
            if piece.piece_type == 'pawn' and (target_position[0] == 0 or target_position[0] == 7):
                board[target_position[0]][target_position[1]] = promote_pawn(piece, 'Q')

            turn = 'white'
            
            # Check for check, checkmate, and stalemate after each move
            if is_checkmate(board, turn):
                print(f"Checkmate! {turn} loses.")
                running = False
            elif is_stalemate(board, turn):
                print("Stalemate! It's a draw.")
                running = False
            elif is_in_check(board, turn):
                print(f"{turn} is in check.")

    # Clear the screen
    screen.fill(BLACK)
    draw_board(screen)
    draw_pieces(screen, board)

    if selected_piece:
        draw_highlight(screen, selected_piece[0], selected_piece[1])
        for move in highlighted_moves:
            draw_highlight(screen, move[0], move[1])

    pygame.display.flip()

    if not running:
        break


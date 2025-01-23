import numpy as np
from ai_training.game_env.utils import *
from ai_training.game_env.tetris_game import *

def calculate_avg_col_height(board, height, width):
        heights = np.zeros(width)
        for y in range(height):
            for x in range(width):
                if board[y][x] != (0,0,0):
                    heights[x] = height - y
                    break
        return np.mean(heights)
def calculate_landing_height(piece, board, valid_space):
        """Calculate the landing height of the piece
            This is the y coordinate where the piece is locked.
        """
        temp_piece = Piece(piece.x, piece.y, piece.shape)
        temp_piece.rotation = piece.rotation
        while valid_space(temp_piece, board):
            temp_piece.y += 1
        return temp_piece.y - 1

def calculate_eroded_cells(piece, board, create_grid, locked_positions, convert_shape_format):
        temp_board = create_grid(locked_positions)
        # lock the piece
        for pos in convert_shape_format(piece):
            p = (pos[0], pos[1])
            if p[1] > -1:
                temp_board[p[1]][p[0]] = piece.color
        # compute the number of full lines.
        lines_cleared = 0
        for i in range(len(temp_board)-1, -1, -1):
            if (0, 0, 0) not in temp_board[i]:
                lines_cleared += 1
        return lines_cleared

def calculate_row_transitions(board):
        transitions = 0
        for row in board:
            for i in range(len(row)-1):
                if (row[i] == (0, 0, 0) and row[i+1] != (0,0,0)) or (row[i] != (0, 0, 0) and row[i+1] == (0,0,0)):
                    transitions+=1
        return transitions
def calculate_column_transitions(board, width, height):
        transitions = 0
        for x in range(width):
            for y in range(height - 1):
                if (board[y][x] == (0, 0, 0) and board[y+1][x] != (0,0,0)) or (board[y][x] != (0, 0, 0) and board[y+1][x] == (0,0,0)):
                    transitions += 1
        return transitions
def calculate_number_of_holes(board, width, height):
        holes = 0
        for x in range(width):
            block_found = False
            for y in range(height):
                if board[y][x] != (0, 0, 0):
                    block_found = True
                if block_found and board[y][x] == (0, 0, 0):
                    holes += 1
        return holes
def calculate_number_of_wells(board, width, height):
        wells = 0
        for x in range(width):
            for y in range(height):
                if board[y][x] != (0, 0, 0):
                # check if it's a well
                    well = True
                    if x > 0 and board[y][x-1] == (0,0,0):
                        well = False
                    if x < width - 1 and board[y][x+1] == (0,0,0):
                        well = False
                    if y < height - 1 and board[y+1][x] == (0,0,0):
                        well = False
                    if well:
                        wells += 1
        return wells
def calculate_hole_depth(board, width, height):
        max_depth = 0
        for x in range(width):
            depth = 0
            block_found = False
            for y in range(height):
                if block_found and board[y][x] == (0, 0, 0):
                    depth +=1
                if board[y][x] != (0, 0, 0):
                    block_found = True
                max_depth = max(max_depth, depth)
        return max_depth
def calculate_rows_with_holes(board, height):
        count = 0
        for row in board:
            hole_found = False
            for cell in row:
                if cell == (0,0,0):
                    hole_found = True
                    break
            if hole_found:
                count += 1
        return count
def calculate_pattern_diversity(board, width, height):
        diversity = 0
        for y in range(height):
            if (board[y].count((0,0,0)) != width) and (board[y].count((0,0,0)) != 0):
                diversity += 1
        return diversity

def extract_dt20_features(piece, board, width, height, valid_space, create_grid, locked_positions, convert_shape_format):
            features = []
            features.append(calculate_landing_height(piece, board, valid_space))
            features.append(calculate_eroded_cells(piece, board, create_grid, locked_positions, convert_shape_format))
            features.append(calculate_row_transitions(board))
            features.append(calculate_column_transitions(board, width, height))
            features.append(calculate_number_of_holes(board, width, height))
            features.append(calculate_number_of_wells(board, width, height))
            features.append(calculate_hole_depth(board, width, height))
            features.append(calculate_rows_with_holes(board, height))
            features.append(calculate_pattern_diversity(board, width, height))
            # Add the RBF features.
            avg_height = calculate_avg_col_height(board, height, width)
            for i in range(5):
                rbf = np.exp(-(avg_height - (height/4 * i))**2 / (2*(height/5)**2))
                features.append(rbf)
            return np.array(features)


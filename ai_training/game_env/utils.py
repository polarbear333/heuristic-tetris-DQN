# utils.py
import pygame
from .config import top_left_x, top_left_y, block_size, row, col, play_height, play_width, s_width, s_height

# utils.py
def draw_grid(surface):
    
    r = g = b = 0
    grid_color = (r, g, b)

    for i in range(row):
        pygame.draw.line(surface, grid_color, (top_left_x, top_left_y + i * block_size),
                         (top_left_x + play_width, top_left_y + i * block_size))
        for j in range(col):
            pygame.draw.line(surface, grid_color, (top_left_x + j * block_size, top_left_y),
                             (top_left_x + j * block_size, top_left_y + play_height))

def convert_shape_format(piece):
    positions = []
    shape_format = piece.shape[piece.rotation % len(piece.shape)]

    for i, line in enumerate(shape_format):
        row = list(line)
        for j, column in enumerate(row):
            if column == '0':
                positions.append((piece.x + j, piece.y + i))

    for i, pos in enumerate(positions):
        positions[i] = (pos[0] - 2, pos[1] - 4)

    return positions

def valid_space(piece, grid):
    # First check if any part of the piece is outside grid boundaries
    formatted_shape = convert_shape_format(piece)
    
    for pos in formatted_shape:
        x, y = pos
        # Check horizontal boundaries (left and right)
        if x < 0 or x >= col:
            return False
        # Check vertical boundary (bottom)
        if y >= row:
            return False
        # Check if position is occupied (only if within grid)
        if y >= 0:
            if grid[y][x] != (0, 0, 0):
                return False
                
    return True
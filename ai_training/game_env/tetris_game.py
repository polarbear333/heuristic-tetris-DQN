# tetris_game.py
import sys
import os
import pygame
import random
from .utils import draw_grid, valid_space, convert_shape_format
from .config import top_left_x, top_left_y, block_size, row, col, play_height, play_width, s_width, s_height

filepath = os.path.abspath('./ai_training/highscore.txt')
fontpath = './ai_training//arcade.ttf'
fontpath_mario = './ai_training/mario.ttf'


S = [['.....',
      '.....',
      '..00.',
      '.00..',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '...0.',
      '.....']]

Z = [['.....',
      '.....',
      '.00..',
      '..00.',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '.0...',
      '.....']]

I = [['.....',
      '..0..',
      '..0..',
      '..0..',
      '..0..'],
     ['.....',
      '0000.',
      '.....',
      '.....',
      '.....']]

O = [['.....',
      '.....',
      '.00..',
      '.00..',
      '.....']]

J = [['.....',
      '.0...',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..00.',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '...0.',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '.00..',
      '.....']]

L = [['.....',
      '...0.',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '..00.',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '.0...',
      '.....'],
     ['.....',
      '.00..',
      '..0..',
      '..0..',
      '.....']]

T = [['.....',
      '..0..',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '..0..',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '..0..',
      '.....']]

shapes = [S, Z, I, O, J, L, T]
shape_colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 165, 0), (0, 0, 255), (128, 0, 128)]

# --- Piece Class ---
class Piece(object):
    def __init__(self, x, y, shape):
        self.x = x
        self.y = y
        self.shape = shape
        self.color = shape_colors[shapes.index(shape)]
        self.rotation = 0

    def get_piece_positions(self):
        positions = []
        shape_format = self.shape[self.rotation % len(self.shape)]

        for i, line in enumerate(shape_format):
            row = list(line)
            for j, column in enumerate(row):
                if column == '0':
                    positions.append((self.x + j, self.y + i))

        for i, pos in enumerate(positions):
            positions[i] = (pos[0] - 2, pos[1] - 4)

        return positions

# --- Tetris Class ---
class Tetris:
    def __init__(self, screen):
        # Initialize pygame.mixer only if a screen is provided (for human rendering)
        if screen is not None:
           pygame.mixer.init()
        self.screen = screen
        self.width = col
        self.height = row
        self.grid = self.create_grid()
        self.current_piece = self.get_shape()
        self.next_piece = self.get_shape()
        self.score = 0
        self.level = 1  # Start at level 1
        self.lines_cleared = 0
        self.fall_time = 0
        self.fall_speed = 350  # Initial speed (milliseconds)
        self.level_time = 0
        self.game_over = False
        self.audio_enabled = False
        # Create the file if it does not exists before loading the highscore
        try:
          open(filepath, 'r').close()
        except FileNotFoundError:
          open(filepath, 'w').close()
        self.last_score = self.get_max_score()
        self.locked_positions = {}

        # Delayed Auto Shift
        self.das_delay = 170  # Delayed Auto Shift initial delay (milliseconds)
        self.das_speed = 50   # How fast pieces move when DAS is active
        self.last_das_time = 0
        self.das_active = False
        self.current_das_direction = None
        
        # Soft drop speed (milliseconds)
        self.soft_drop_speed = 50
        self.normal_fall_speed = 350
        self.fall_speed = self.normal_fall_speed
        
        # Button press tracking
        self.keys_pressed = set()
        self.last_move_time = 0
        self.clock = pygame.time.Clock()
        
        self.move_sound = None
        self.rotate_sound = None
        self.clear_sound = None
        self.gameover_sound = None

        if screen is not None:
            try:
                self.move_sound = pygame.mixer.Sound("ai_training/sounds/move.mp3")
                self.rotate_sound = pygame.mixer.Sound("ai_training/sounds/rotate.mp3")
                self.clear_sound = pygame.mixer.Sound("ai_training/sounds/success.wav")
                self.gameover_sound = pygame.mixer.Sound("ai_training/sounds/gameover.wav")
            except pygame.error as e:
                print(f"Error loading sounds: {e}")
                self.move_sound = None
                self.rotate_sound = None
                self.clear_sound = None
                self.gameover_sound = None
        
    def set_audio(self, enabled=True):
        """Enable or disable game audio"""
        self.audio_enabled = enabled

    def play_sound(self, sound):
        """Play sound if audio is enabled"""
        if self.audio_enabled and sound is not None:
            sound.play()

    def create_grid(self, locked_pos={}):
        grid = [[(0, 0, 0) for _ in range(col)] for _ in range(row)]
        for y in range(row):
            for x in range(col):
                if (x, y) in locked_pos:
                    color = locked_pos[(x, y)]
                    grid[y][x] = color
        return grid

    def valid_space(self, piece, grid):
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
                if self.grid[y][x] != (0, 0, 0):
                    return False
                    
        return True

    def get_shape(self):
        return Piece(5, 0, random.choice(shapes))

    def check_lost(self, positions):
        for pos in positions:
            x, y = pos
            # Check if any piece is at or above the top of the grid
            # or if any piece is outside the valid x bounds
            if y < 1 or x < 0 or x >= col:
                return True
        return False
    
    def get_locked_positions(self): 
        locked_positions = {}
        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                if cell != (0, 0, 0):
                    locked_positions[(x, y)] = cell
        return locked_positions

    def clear_rows(self):
        # Find all complete rows
        complete_rows = []
        for i in range(len(self.grid)-1, -1, -1):
            if (0, 0, 0) not in self.grid[i]:
                complete_rows.append(i)
        
        if not complete_rows:
            return 0
            
        # Remove all blocks in complete rows from locked positions
        for row in complete_rows:
            for j in range(len(self.grid[0])):
                try:
                    del self.locked_positions[(j, row)]
                except:
                    continue

        # Move down all blocks above the uppermost cleared row
        highest_row = min(complete_rows)  # Get the highest row that was cleared
        positions_to_move = {}
        
        # Sort positions from bottom to top for proper movement
        sorted_positions = sorted(list(self.locked_positions.items()), 
                                key=lambda x: x[0][1], reverse=True)
        
        # Calculate how many rows down each piece should move
        for pos, color in sorted_positions:
            x, y = pos
            if y < highest_row:  # If the position is above any cleared row
                # Count how many cleared rows are below this position
                shift = sum(1 for row in complete_rows if row > y)
                if shift > 0:
                    positions_to_move[(x, y + shift)] = color
                    del self.locked_positions[pos]
        
        # Update locked positions with new positions
        self.locked_positions.update(positions_to_move)
        
        # Update the grid
        self.grid = self.create_grid(self.locked_positions)
        
        return len(complete_rows)
    def update(self):
        current_time = pygame.time.get_ticks()
        
        # Handle automatic falling
        if current_time - self.fall_time > self.fall_speed:
            self.fall_time = current_time
            self.current_piece.y += 1
            
            # Check if piece is in valid position
            if not valid_space(self.current_piece, self.grid):
                self.current_piece.y -= 1
                self.lock_piece()
                return True
                
        return False

    def draw_next_shape(self, piece, surface):
        font = pygame.font.Font(fontpath, 30)
        label = font.render('Next shape', 1, (255, 255, 255))
        start_x = top_left_x + play_width + 50
        start_y = top_left_y + (play_height / 2 - 100)
        shape_format = piece.shape[piece.rotation % len(piece.shape)]

        for i, line in enumerate(shape_format):
            row = list(line)
            for j, column in enumerate(row):
                if column == '0':
                    pygame.draw.rect(surface, piece.color, (start_x + j*block_size, start_y + i*block_size, block_size, block_size), 0)

        surface.blit(label, (start_x, start_y - 30))
    
    def draw_play_again_button(self, surface):
        font = pygame.font.Font(fontpath, 30)
        label = font.render("Play Again (Right Shift)", 1, (255, 255, 255))
        button_rect = pygame.Rect(top_left_x + play_width / 2 - (label.get_width() / 2) - 10,
                                 top_left_y + play_height / 2 + 50, label.get_width() + 20, 50)
        pygame.draw.rect(surface, (0, 0, 0), button_rect)
        pygame.draw.rect(surface, (255, 255, 255), button_rect, 2)  # Draw a white border
        surface.blit(label, (button_rect.x + 10, button_rect.y + 10))
        return button_rect

    def reset_game(self):
        self.grid = self.create_grid()
        self.current_piece = self.get_shape()
        self.next_piece = self.get_shape()
        self.score = 0
        self.level = 1
        self.lines_cleared = 0
        self.fall_time = 0
        self.fall_speed = 350
        self.level_time = 0
        self.game_over = False
        self.locked_positions = {}
        self.last_move_time = 0
        self.das_active = False
        self.current_das_direction = None
        self.soft_drop_speed = 50
        self.normal_fall_speed = 350
        self.fall_speed = self.normal_fall_speed
        self.keys_pressed = set()
    
    def render(self, screen):
        screen.fill((1, 50, 32))
        pygame.font.init()
        font = pygame.font.Font(fontpath_mario, 65)
        label = font.render('AI TETRIS', 1, (255, 255, 255))
        screen.blit(label, ((top_left_x + play_width / 2) - (label.get_width() / 2), 30))


        # Draw the locked pieces from the grid
        for i in range(row):
            for j in range(col):
                pygame.draw.rect(screen, self.grid[i][j],
                               (top_left_x + j * block_size, top_left_y + i * block_size, block_size, block_size), 0)

        # Draw the current piece
        if self.current_piece:
            piece_pos = convert_shape_format(self.current_piece)
            for pos in piece_pos:
                x, y = pos
                if y >= 0:  # Only draw if the piece is on screen
                    pygame.draw.rect(screen, self.current_piece.color,
                                   (top_left_x + x * block_size, top_left_y + y * block_size, block_size, block_size), 0)

        # Draw ghost piece
        ghost_piece = self.get_ghost_piece_position()
        if ghost_piece:
            ghost_pos = convert_shape_format(ghost_piece)
            for pos in ghost_pos:
                x, y = pos
                if y >= 0:
                    pygame.draw.rect(screen, (0, 255, 255),  # Gray color for ghost piece
                                   (top_left_x + x * block_size, top_left_y + y * block_size, 
                                    block_size, block_size), 1)  # Draw outline only

        # Draw the grid lines
        draw_grid(screen)
        
        # Draw the border
        pygame.draw.rect(screen, (1, 50, 32), (top_left_x, top_left_y, play_width, play_height), 4)

        # Draw next shape
        self.draw_next_shape(self.next_piece, screen)

        # Draw scores
        font = pygame.font.Font(fontpath, 30)
        label = font.render('SCORE   ' + str(self.score), 1, (255, 255, 255))
        start_x = top_left_x + play_width + 50
        start_y = top_left_y + (play_height / 2 - 100)
        screen.blit(label, (start_x, start_y + 200))

        label_hi = font.render('HIGHSCORE   ' + str(self.last_score), 1, (255, 255, 255))
        start_x_hi = top_left_x - 240
        start_y_hi = top_left_y + 200
        screen.blit(label_hi, (start_x_hi + 20, start_y_hi + 200))

        # Level
        label_level = font.render('LEVEL   ' + str(self.level), 1, (255, 255, 255))
        screen.blit(label_level, (start_x_hi + 20, start_y_hi + 100))

        # Speed (convert from milliseconds to blocks per second)
        speed_blocks_per_second = 1000 / self.fall_speed
        label_speed = font.render('SPEED   {:.1f}'.format(speed_blocks_per_second), 1, (255, 255, 255))
        screen.blit(label_speed, (start_x_hi + 20, start_y_hi + 150))

        


    def draw_text_middle(self, text, size, color, surface):
        pygame.font.init()  # Initialize the font module
        font = pygame.font.Font(fontpath, size)
        label = font.render(text, 1, color)
        surface.blit(label, (top_left_x + play_width / 2 - (label.get_width() / 2),
                             top_left_y + play_height / 2 - (label.get_height() / 2)))

    def update_score(self, new_score):
        score = self.get_max_score()
        with open(filepath, 'w') as file:
            if new_score > score:
                file.write(str(new_score))
            else:
                file.write(str(score))

    def get_max_score(self):
        try:
            if not os.path.exists(filepath): # If the file does not exist create it.
              open(filepath, 'w').close()
            with open(filepath, 'r') as file:
                lines = file.readlines()
                if lines:  # Check if the list is not empty
                    score = int(lines[0].strip())
                else:
                    score = 0  # Return default score if file is empty
        except FileNotFoundError:
            # Create the file if it does not exist and return 0
            open(filepath, 'w').close()
            score = 0
        return score
    
    def get_ghost_piece_position(self):
        """Returns the position where the current piece would land"""
        if not self.current_piece:
            return None
            
        ghost_piece = Piece(self.current_piece.x, self.current_piece.y, self.current_piece.shape)
        ghost_piece.rotation = self.current_piece.rotation
        
        # Drop the ghost piece until it hits something
        while valid_space(ghost_piece, self.grid):
            ghost_piece.y += 1
        ghost_piece.y -= 1
        
        return ghost_piece
    
    def hard_drop(self):
        """Instantly drops the piece to the bottom and locks it"""
        while valid_space(self.current_piece, self.grid):
            self.current_piece.y += 1
        self.current_piece.y -= 1
        self.lock_piece()
        
    def lock_piece(self):
        """Locks the current piece in place and handles piece changing"""
        # Add piece to locked positions
        for pos in convert_shape_format(self.current_piece):
            p = (pos[0], pos[1])
            if p[1] > -1:
                self.locked_positions[p] = self.current_piece.color
        
        # Update grid
        self.grid = self.create_grid(self.locked_positions)
        
        # Check for completed rows
        rows_cleared = self.clear_rows()
        if rows_cleared > 0:
            self.score += rows_cleared * 10
            self.lines_cleared += rows_cleared
            if self.clear_sound:
                self.clear_sound.play()
            
            if self.lines_cleared >= 5:
                self.level += 1
                self.lines_cleared -= 5
                self.normal_fall_speed = max(100, self.normal_fall_speed * 0.95)
        
        # Check for game over
        if self.check_lost(self.locked_positions):
            self.game_over = True
            if self.gameover_sound:
                self.gameover_sound.play()
            return True
        
        # Get next piece
        self.current_piece = self.next_piece
        self.next_piece = self.get_shape()
        self.update_score(self.score)
        return False

    def handle_input(self, events):
        current_time = pygame.time.get_ticks()
        
        for event in events:
            if event.type == pygame.KEYDOWN:
                self.keys_pressed.add(event.key)
                
                # Immediate movement on key press
                if event.key == pygame.K_LEFT:
                    # Store current position in case we need to revert
                    original_x = self.current_piece.x
                    self.current_piece.x -= 1
                    if not valid_space(self.current_piece, self.grid):
                        self.current_piece.x = original_x
                        self.current_das_direction = None
                        self.das_active = False
                    else:
                        self.last_das_time = current_time
                        self.current_das_direction = pygame.K_LEFT
                        if self.move_sound:
                            self.move_sound.play()
                
                elif event.key == pygame.K_RIGHT:
                    # Store current position in case we need to revert
                    original_x = self.current_piece.x
                    self.current_piece.x += 1
                    if not valid_space(self.current_piece, self.grid):
                        self.current_piece.x = original_x
                        self.current_das_direction = None
                        self.das_active = False
                    else:
                        self.last_das_time = current_time
                        self.current_das_direction = pygame.K_RIGHT
                        if self.move_sound:
                            self.move_sound.play()
                
                elif event.key == pygame.K_DOWN:
                    self.fall_speed = self.soft_drop_speed
                
                elif event.key == pygame.K_UP:
                    old_rotation = self.current_piece.rotation
                    self.current_piece.rotation = (self.current_piece.rotation + 1) % len(self.current_piece.shape)
                    if not valid_space(self.current_piece, self.grid):
                        self.current_piece.rotation = old_rotation
                    elif self.rotate_sound:
                        self.rotate_sound.play()
                
                elif event.key == pygame.K_SPACE:
                    self.hard_drop()
            
            elif event.type == pygame.KEYUP:
                if event.key in self.keys_pressed:
                    self.keys_pressed.remove(event.key)
                if event.key == pygame.K_DOWN:
                    self.fall_speed = self.normal_fall_speed
                if event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                    self.current_das_direction = None
                    self.das_active = False

        # Handle DAS (Delayed Auto Shift)
        if self.current_das_direction:
            if current_time - self.last_das_time > self.das_delay or self.das_active:
                self.das_active = True
                if current_time - self.last_move_time > self.das_speed:
                    original_x = self.current_piece.x
                    
                    if self.current_das_direction == pygame.K_LEFT:
                        self.current_piece.x -= 1
                    elif self.current_das_direction == pygame.K_RIGHT:
                        self.current_piece.x += 1
                    
                    if not valid_space(self.current_piece, self.grid):
                        self.current_piece.x = original_x
                        self.current_das_direction = None
                        self.das_active = False
                    else:
                        self.last_move_time = current_time
                        if self.move_sound:
                            self.move_sound.play()

def main(game):
    run = True
    while run:
        game.grid = game.create_grid(game.get_locked_positions())

        # Handle events
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        # Handle input with new system
        game.handle_input(events)

        # Update game state
        if not game.game_over:
            game.update()

        game.render(game.screen)
        pygame.display.update()
        game.clock.tick(60)

        if game.game_over:
            game.draw_text_middle('Game Over!', 40, (255, 255, 255), game.screen)
            button_rect = game.draw_play_again_button(game.screen)
            pygame.display.update()

            waiting_for_input = True
            while waiting_for_input:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        waiting_for_input = False
                        run = False
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_RSHIFT:
                            game.reset_game()
                            waiting_for_input = False
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if button_rect.collidepoint(event.pos):
                            game.reset_game()
                            waiting_for_input = False

    pygame.quit()

def main_menu(window, game):
    run = True
    while run:
        window.fill((42, 82, 190))
        game.draw_text_middle('Press any key to begin', 50, (255, 255, 255), window)
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                main(game)
    pygame.quit()

if __name__ == "__main__":
    win = pygame.display.set_mode((s_width, s_height))
    game = Tetris(win)
    main_menu(win, game)
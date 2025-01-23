import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import sys
import os
sys.path.append(os.path.abspath('./game_env'))
from ai_training.game_env.tetris_game import Tetris, main_menu, s_width, s_height, Piece, convert_shape_format
from ai_training.rl_agent.environment.features import *


class TetrisEnv(gym.Env):
    def __init__(self, agent=None): # add an agent argument to the constructor.
        super().__init__()
        self.win = None
        self.game = None
        self.action_space = spaces.Discrete(5)  # 5 discrete actions (left, right, rotate, down, hard_drop)
        self.observation_space = spaces.Dict({
                "grid": spaces.Box(low=0, high=1, shape=(20, 10), dtype=np.float32),
                "features": spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)
        })
        self.action_list = ["left", "right", "rotate", "down", "hard_drop"]
        self.valid_space = lambda piece, grid: self.game.valid_space(piece, grid)
        self.create_grid = lambda locked: self.game.create_grid(locked)
        self._first_reset = True
        self.agent = agent  # set the agent here.
    def set_audio(self, enabled=True):
         if self.game:
           self.game.set_audio(enabled)
    def reset(self, visualize=True, seed=None, options=None):
        super().reset(seed=seed)
        if self._first_reset:
            if visualize:
                pygame.init()
                self.win = pygame.display.set_mode((s_width, s_height))
            self.game = Tetris(self.win) # initialize the game only when reset is called.
            self._first_reset = False # only runs this code once.
        self.game.reset_game()
        state = self.get_state_representation()
        info = {}
        return state, info
    def step(self, action_index):
        action = self.action_list[action_index]
        current_state = self.get_state_representation()

        if action == "left":
            self.game.current_piece.x -= 1
            if not self.game.valid_space(self.game.current_piece, self.game.grid):
                self.game.current_piece.x += 1

        elif action == "right":
            self.game.current_piece.x += 1
            if not self.game.valid_space(self.game.current_piece, self.game.grid):
                self.game.current_piece.x -= 1

        elif action == "rotate":
            old_rotation = self.game.current_piece.rotation
            self.game.current_piece.rotation = (self.game.current_piece.rotation + 1) % len(self.game.current_piece.shape)
            if not self.game.valid_space(self.game.current_piece, self.game.grid):
                self.game.current_piece.rotation = old_rotation

        elif action == "down":
            self.game.current_piece.y += 1
            if not self.game.valid_space(self.game.current_piece, self.game.grid):
                self.game.current_piece.y -= 1
                self.game.lock_piece()

        elif action == "hard_drop":
            self.game.hard_drop()

        # Ensure the game state is updated correctly after each action
        self.game.update()

        next_state = self.get_state_representation(action)
        reward = self.calculate_reward(current_state, action, next_state)
        terminated = self.game.game_over
        truncated = False
        info = {}
        return next_state, reward, terminated, truncated, info

    def calculate_reward(self, current_state, action, next_state):
        reward = 0
        
        # 1. Primary Rewards
        cleared_rows = self.game.clear_rows()
        if cleared_rows > 0:
            reward += cleared_rows * 1000  # Major reward for line clears
        reward += 5  # Survival bonus
        
        if self.game.game_over:
            reward -= 100  # Moderate penalty

        # 2. Heuristic-based Shaping (Normalized)
        action_index = self.action_list.index(action)
        
        # Get normalized scores [0,1]
        v_s = self.agent.get_heuristic_score(current_state, action_index)  # 0-1
        v_s_prime = self.agent.get_heuristic_score(next_state, action_index)  # 0-1
        
        # Scale to match reward magnitudes
        heuristic_scale = 100  # Matches line clear rewards
        lambda_value = 0.3  # Reduced from 0.5 for stability
        
        # Shaping components
        future_bonus = v_s_prime * heuristic_scale * 0.2  # Encourage good next states
        improvement_bonus = (v_s_prime - v_s) * heuristic_scale * lambda_value
        
        reward += future_bonus + improvement_bonus

        return reward
    def close(self):
      if self.win:
          pygame.quit()

    def get_grid_representation(self):
        grid_array = np.array(self.game.grid) # convert to numpy array
        binary_grid = np.any(grid_array != [0, 0, 0], axis=2).astype(int) # check if cells are not empty
        return torch.tensor(binary_grid, dtype=torch.float32) # convert to tensor

    def get_state_representation(self, action=None):
        """Returns tuple of (grid_tensor, features_tensor)"""
        # Get current grid state
        grid_tensor = self.get_grid_representation()  # Shape [20, 10]

        if action is not None:
            # Create temporary copies for action simulation
            temp_locked_positions = dict(self.game.locked_positions)
            piece_clone = Piece(self.game.current_piece.x, 
                            self.game.current_piece.y,
                            self.game.current_piece.shape)
            piece_clone.rotation = self.game.current_piece.rotation

            # Simulate the action
            if action == "left":
                piece_clone.x -= 1
            elif action == "right":
                piece_clone.x += 1
            elif action == "rotate":
                piece_clone.rotation = (piece_clone.rotation + 1) % len(piece_clone.shape)
            elif action == "down":
                piece_clone.y += 1

            # Validate and adjust position
            temp_grid = self.game.create_grid(temp_locked_positions)
            if not self.game.valid_space(piece_clone, temp_grid):
                piece_clone = self.game.current_piece
            else:
                if action == "down":
                    # Simulate dropping to bottom
                    while self.game.valid_space(piece_clone, temp_grid):
                        piece_clone.y += 1
                    piece_clone.y -= 1
                    
                    # Update temp locked positions
                    for pos in convert_shape_format(piece_clone):
                        p = (pos[0], pos[1])
                        if p[1] > -1:
                            temp_locked_positions[p] = self.game.current_piece.color
                    temp_grid = self.game.create_grid(temp_locked_positions)

            # Extract features from simulated state
            features = extract_dt20_features(
                piece_clone, temp_grid, self.game.width, self.game.height,
                self.game.valid_space, self.game.create_grid,
                temp_locked_positions, convert_shape_format
            )
        else:
            # Extract features from current state
            features = extract_dt20_features(
                self.game.current_piece, self.game.grid, self.game.width, self.game.height,
                self.game.valid_space, self.game.create_grid,
                self.game.locked_positions, convert_shape_format
            )

        # Normalize features
        normalized_features = (features - np.mean(features)) / np.std(features) if np.std(features) != 0 else features
        features_tensor = torch.tensor(normalized_features, dtype=torch.float32)

        return (grid_tensor, features_tensor)
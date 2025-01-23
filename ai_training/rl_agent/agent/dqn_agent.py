import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from ai_training.rl_agent.models.q_network import QNetwork
from ai_training.rl_agent.agent.replay_buffer import PrioritizedReplayBuffer
from ai_training.rl_agent.utils.training_utils import linear_epsilon_decay, tensor_to_numpy
from ai_training.game_env.tetris_game import shapes, Piece
from ai_training.game_env.utils import convert_shape_format, valid_space
from ai_training.rl_agent.environment.tetris_env import TetrisEnv
from ai_training.rl_agent.environment.features import extract_dt20_features
import warnings
class DQNAgent:
    def __init__(self, row, col, feature_size, conv_channels,hidden_size, output_size,
                 learning_rate=0.001, gamma=0.99, buffer_size = 10000, batch_size=32, tau=0.005,
                 epsilon_start = 1.0, epsilon_end=0.01, num_episodes = 1000, epsilon_heuristic=0.5,
                 shape = shapes[0], env = None, config=None):
        self.row = row
        self.col = col
        self.env = env
        self.feature_size = feature_size
        self.output_size = output_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = QNetwork(
            grid_shape=(row, col),
            feature_size=feature_size,
            output_size=output_size,
            conv_channels=conv_channels,
            hidden_size=hidden_size
        ).to(self.device)
        
        self.target_network = QNetwork(
            grid_shape=(row, col),
            feature_size=feature_size,
            output_size=output_size,
            conv_channels=conv_channels,
            hidden_size=hidden_size
        ).to(self.device)

        self.target_network.load_state_dict(self.q_network.state_dict())  # Initialize target with the same weights
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.buffer = PrioritizedReplayBuffer(buffer_size)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.num_episodes = num_episodes
        self.epsilon_heuristic = epsilon_heuristic
        self.temperature = 1.0  # Initial temperature
        self.temperature_decay = 0.995  # Decay rate per episode
        self.loss_fn = nn.MSELoss()
        self.epsilon_decay_function = linear_epsilon_decay(epsilon_start, epsilon_end, num_episodes)
        self.shape = shape # set up shape parameter here.
        self.extract_dt20_features = extract_dt20_features 
        self.config = config


    def choose_action(self, state):
        """Select action using hybrid heuristic/Q-network policy"""
        grid, features = state
    
    # Convert to tensors properly
        grid_tensor = torch.as_tensor(grid, dtype=torch.float32).unsqueeze(0).to(self.device)
        features_tensor = torch.as_tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Heuristic-guided exploration (DT-20 policy)
        if random.random() < self.epsilon_heuristic:
            heuristic_scores = np.zeros(self.output_size)
            
            for action_idx in range(self.output_size):
                # Simulate action without affecting real game state
                temp_piece = Piece(self.env.game.current_piece.x,
                                self.env.game.current_piece.y,
                                self.env.game.current_piece.shape)
                temp_piece.rotation = self.env.game.current_piece.rotation
                temp_locked = dict(self.env.game.locked_positions)
                
                # Apply simulated action
                if self.env.action_list[action_idx] == "left":
                    temp_piece.x -= 1
                elif self.env.action_list[action_idx] == "right":
                    temp_piece.x += 1
                elif self.env.action_list[action_idx] == "rotate":
                    temp_piece.rotation = (temp_piece.rotation + 1) % len(temp_piece.shape)
                
                # Calculate heuristic score for simulated state
                if self.env.game.valid_space(temp_piece, self.env.game.grid):
                    heuristic_scores[action_idx] = self.get_heuristic_score(
                        (grid.numpy(), features.numpy()), action_idx
                    )
                else:
                    heuristic_scores[action_idx] = -np.inf  # Invalid action

            # Boltzmann exploration over heuristic scores
            scaled_scores = heuristic_scores / self.temperature
            exp_scores = np.exp(scaled_scores - np.max(scaled_scores))
            probs = exp_scores / np.sum(exp_scores)
            action = np.random.choice(self.output_size, p=probs)

        # Q-network policy
        else:
            with torch.no_grad():
                q_values = self.q_network(grid_tensor, features_tensor).squeeze(0)
                
            # Epsilon-greedy exploration
            if random.random() < self.epsilon:
                action = random.randint(0, self.output_size-1)
            else:
                action = torch.argmax(q_values).item()

        return action
    
    
    def get_heuristic_score(self, state, action_index):
        grid, features = state  # Unpack the state tuple
        grid_tensor = torch.as_tensor(grid, dtype=torch.float32)
        features_tensor = torch.as_tensor(features, dtype=torch.float32)
        x, y = self.env.game.current_piece.x, self.env.game.current_piece.y
        piece_clone = Piece(x, y, self.shape)
        piece_clone.rotation = self.env.game.current_piece.rotation
        action_list = ["left", "right", "rotate", "down", "hard_drop"]
        action = action_list[action_index]

        with torch.no_grad():
            # Apply action simulation
            if action == "left":
                piece_clone.x -= 1
            elif action == "right":
                piece_clone.x += 1
            elif action == "rotate":
                piece_clone.rotation = (piece_clone.rotation + 1) % len(piece_clone.shape)
            elif action == "down":
                piece_clone.y += 1

        temp_grid = self.env.game.create_grid(self.env.game.locked_positions)
        if not valid_space(piece_clone, temp_grid):
            # Normalized penalty for invalid actions
            return 0.0  # Now in same scale as valid actions
        else:
            # Calculate features and raw score
            features = self.extract_dt20_features(
                piece_clone, temp_grid, self.col, self.row,
                self.env.game.valid_space, self.env.game.create_grid,
                self.env.game.locked_positions, convert_shape_format
            )
            dt_20_weights = np.array([
                -2.68, 1.38, -2.41, -6.32, 2.03, 
                -2.71, -0.43, -9.48, 0.89, 0.5, 0.3, 0.2, 0.1, 0.05
            ], dtype=np.float32)
            raw_score = np.dot(features, dt_20_weights)

            # Normalization parameters (empirical values from paper)
            min_valid = -1000  # Estimated minimum valid score
            max_valid = -242.55   # Estimated maximum valid score
            
            # Clip and normalize valid scores
            clipped_score = np.clip(raw_score, min_valid, max_valid)
            normalized = (clipped_score - min_valid) / (max_valid - min_valid)
            return float(normalized)  # Returns 0-1 for valid actions



    def learn(self):
        if len(self.buffer.buffer) < self.batch_size:
            return None, None, None  # Not enough samples

        # Sample from prioritized replay buffer
        (current_grids, current_features, 
        actions, rewards, 
        next_grids, next_features, 
        dones, indices, 
        weights) = self.buffer.sample(self.batch_size)

        # Convert to tensors with proper device placement
        current_grids = torch.as_tensor(current_grids, dtype=torch.float32, device=self.device)
        current_features = torch.as_tensor(current_features, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_grids = torch.as_tensor(next_grids, dtype=torch.float32, device=self.device)
        next_features = torch.as_tensor(next_features, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.bool, device=self.device)
        weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device)

        # Current Q-values for taken actions
        current_q_values = self.q_network(current_grids, current_features)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values with double DQN
        with torch.no_grad():
            # Use main network for action selection
            next_q_values = self.q_network(next_grids, next_features)
            next_actions = next_q_values.argmax(1)
            
            # Use target network for evaluation
            next_target_values = self.target_network(next_grids, next_features)
            next_q = next_target_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            # Calculate target with termination mask
            target_q = rewards + (self.gamma * next_q * ~dones)

        # Compute importance-weighted MSE loss
        td_errors = torch.abs(target_q - current_q)
        loss = (weights * F.mse_loss(current_q, target_q, reduction='none')).mean()

        # Optimize the Q-network
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        if self.config.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config["grad_clip"])
        
        self.optimizer.step()

        # Update priorities in replay buffer
        with torch.no_grad():
            new_priorities = td_errors.cpu().numpy() + 1e-6  # Add small epsilon
        self.buffer.update_priorities(indices, new_priorities)

        # Soft update target network
        self.update_target_network()

        return (
            loss.item(), 
            current_q.detach().mean().item(),
            target_q.detach().mean().item()
        )

    def update_target_network(self):
         for target_param, q_param in zip(self.target_network.parameters(), self.q_network.parameters()):
           target_param.data.copy_(self.tau * q_param.data + (1 - self.tau) * target_param.data)

    def update_epsilon(self, episode):
       self.epsilon = self.epsilon_decay_function(episode)
       self.temperature = max(self.temperature * self.temperature_decay, 0.1)

    def save_checkpoint(self, path, episode):
        """Save full training state to resume later"""
        checkpoint = {
            'episode': episode,
            'q_state_dict': self.q_network.state_dict(),
            'target_state_dict': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'buffer': self.buffer.buffer,  # Save replay buffer content
            'epsilon': self.epsilon,
            'temperature': self.temperature,
            'epsilon_heuristic': self.epsilon_heuristic
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """Load training state from checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load neural networks
        self.q_network.load_state_dict(checkpoint['q_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_state_dict'])
        
        # Load optimizer
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        # Load replay buffer
        self.buffer.buffer = checkpoint['buffer']
        
        # Load training states
        self.epsilon = checkpoint['epsilon']
        self.temperature = checkpoint['temperature']
        self.epsilon_heuristic = checkpoint['epsilon_heuristic']
        
        return checkpoint['episode']  # Return last episode to resume
    
    def save_model(self, path):
      torch.save(self.q_network.state_dict(), path)
    def load_model(self, path):
      self.q_network.load_state_dict(torch.load(path))
      self.target_network.load_state_dict(self.q_network.state_dict())

    
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ai_training.game_env.config import row, col
from ai_training.game_env.tetris_game import shapes



def get_config():
    config = {
        # Network Architecture
        "conv_channels": 32,        # Number of CNN filters
        "hidden_size": 256,         # Size of dense layer after CNN
        "row": row,                 # From game config (usually 20)
        "col": col,                 # From game config (usually 10)
        "shape": shapes[0],         # Default tetromino shape
        
        # Training Parameters
        "learning_rate": 0.0001,
        "gamma": 0.99,
        "tau": 0.005,
        
        # Exploration
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_heuristic": 0.7,
        
        # Experience Management
        "buffer_size": 1000000,
        "batch_size": 512,
        
        # Training Schedule
        "num_episodes": 10000,
        "max_steps": 1000,
        "checkpoint_interval": 50,  # Save every 50 episodes
        "resume": False,  
        "log_interval": 100,
        "save_interval": 100,
        "grad_clip": 10.0,
        
        # Paths
        "model_path": "ai_training/rl_agent/trained_models"
    }

    # Create model directory if needed
    if not os.path.exists(config["model_path"]):
        os.makedirs(config["model_path"])

    return config
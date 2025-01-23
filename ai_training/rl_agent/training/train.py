import gymnasium as gym
import torch
import os
import sys
import random
import numpy as np
from collections import defaultdict
from ai_training.rl_agent.environment.tetris_env import TetrisEnv
from ai_training.rl_agent.agent.dqn_agent import DQNAgent
from ai_training.rl_agent.utils.config import get_config  # Import the get_config function
from ai_training.rl_agent.utils.logger import Logger # Import the Logger class
from ai_training.game_env.tetris_game import shapes
from ai_training.rl_agent.utils.training_utils import calculate_max_height, print_training_metrics


import threading
sys.path.append(os.path.abspath('./game_env'))
from ai_training.rl_agent.utils.training_utils import tensor_to_numpy

def train_logic(agent, env, logger, config, visualize):
    num_episodes = config["num_episodes"]
    max_steps = config["max_steps"]
    start_episode = 0

    # Check for existing checkpoint
    if config["resume"]:
        checkpoint_path = os.path.join(config["model_path"], "checkpoint_latest.pth")
        if os.path.exists(checkpoint_path):
            start_episode = agent.load_checkpoint(checkpoint_path) + 1
            print(f"⏩ Resuming training from episode {start_episode}")
            print(f"   Initial epsilon: {agent.epsilon:.2f}")
            print(f"   Buffer size: {len(agent.buffer.buffer)}")

    for episode in range(start_episode, num_episodes):
        state, info = env.reset(visualize)
        done = False
        total_reward = 0
        step_count = 0
        metrics = {
            'total_reward': 0,
            'steps': 0,
            'lines_cleared': 0,
            'max_height': 0,
            'action_counts': defaultdict(int)
        }

        while not done and step_count < max_steps:
            # Get action and step through environment
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store experience with proper state components
            agent.buffer.add((
                state[0].numpy(),  # Grid
                state[1].numpy(),  # Features
                action,
                reward,
                next_state[0].numpy() if next_state is not None else None,
                next_state[1].numpy() if next_state is not None else None,
                done
            ))

            # Learn from experience
            loss, q_values, targets = agent.learn()

            # Update metrics
            metrics['total_reward'] += reward
            metrics['steps'] += 1
            metrics['action_counts'][env.action_list[action]] += 1
            metrics['lines_cleared'] += getattr(env.game, 'lines_cleared_this_step', 0)
            metrics['max_height'] = max(metrics['max_height'], 
                                      calculate_max_height(env.game.grid))

            # Update state
            state = next_state
            step_count += 1

        # Episode post-processing
        agent.update_epsilon(episode)
        avg_reward = metrics['total_reward'] / max(metrics['steps'], 1)
        
        # Save checkpoint periodically
        if (episode + 1) % config["checkpoint_interval"] == 0:
            checkpoint_path = os.path.join(
                config["model_path"], 
                f"checkpoint_ep{episode+1}.pth"
            )
            agent.save_checkpoint(checkpoint_path, episode)
            agent.save_checkpoint(  # Always keep latest
                os.path.join(config["model_path"], "checkpoint_latest.pth"), 
                episode
            )
            print(f"💾 Saved checkpoint for episode {episode+1}")

        # Logging and model saving
        if (episode + 1) % config["log_interval"] == 0:
            logger.log(
                episode=episode + 1,
                avg_reward=avg_reward,
                epsilon=agent.epsilon,
                steps=metrics['steps'],
                lines_cleared=metrics['lines_cleared'],
                max_height=metrics['max_height'],
                action_counts=dict(metrics['action_counts'])
            )

        if (episode + 1) % config["save_interval"] == 0:
            model_path = os.path.join(
                config["model_path"], 
                f"model_ep{episode+1}.pth"
            )
            agent.save_model(model_path)
            print(f"🎯 Saved model for episode {episode+1}")

        # Print console output
        print_training_metrics(
            episode + 1, avg_reward, agent.epsilon, metrics['steps'],
            metrics['total_reward'], metrics['lines_cleared'],
            metrics['max_height'], dict(metrics['action_counts']),
            len(agent.buffer.buffer), num_episodes
        )

    # Final save
    agent.save_model(os.path.join(config["model_path"], "model_final.pth"))
    print("🏁 Training complete! Saved final model.")


def train(visualize=True):
    start_episode = 0
    # get configuration parameters
    config = get_config()

    # initialize logger
    logger = Logger(config)

    # check cuda
    if torch.cuda.is_available():
       print('Using GPU')
    else:
        print("using CPU")


    # Initialize the environment
    env = TetrisEnv()
    state, info = env.reset(visualize)  # initialize the game before creating DQNAgent

    agent = DQNAgent(
        row=env.game.height,
        col=env.game.width,
        feature_size=14,  # DT-20 features count (9 base + 5 RBF)
        conv_channels=config["conv_channels"],
        hidden_size=config["hidden_size"],  # From config
        output_size=env.action_space.n,
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        buffer_size=config["buffer_size"],
        batch_size=config["batch_size"],
        tau=config["tau"],
        epsilon_start=config["epsilon_start"],
        epsilon_end=config["epsilon_end"],
        num_episodes=config["num_episodes"],
        epsilon_heuristic=config["epsilon_heuristic"],
        shape=config["shape"],  # From config
        config=config,
        env=env
    )


    # Load Model if it exists
    model_path = config["model_path"]
    if os.path.exists(model_path):
        saved_models = [f for f in os.listdir(model_path) if f.startswith("model_episode_") and f.endswith(".pth")]
        if saved_models:
            # Find the latest model based on the episode number
            saved_models.sort(key=lambda f: int(f.split("_")[-1].split(".")[0]))
            latest_model = saved_models[-1]
            latest_model_path = os.path.join(model_path, latest_model)
            agent.load_model(latest_model_path)
            print(f"Loaded existing model from {latest_model_path}")
        else:
          print("No Existing Model was found")
    else:
      print("Model directory was not found")

    env.agent = agent
    env.set_audio(visualize)

    training_thread = threading.Thread(target=train_logic, args=(agent, env, logger, config, visualize))
    training_thread.start()
    running = True
    if visualize:
        import pygame
        while running:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    running = False
                # Pass events to game environment
                env.game.handle_input(events)

            # Render game state
            env.game.render(env.win)
            pygame.display.update()

        env.close()
    training_thread.join()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train a DQN agent for Tetris.')
    parser.add_argument('--no-visual', action='store_true', help='Disable visualization')
    args = parser.parse_args()

    train(visualize=not args.no_visual)
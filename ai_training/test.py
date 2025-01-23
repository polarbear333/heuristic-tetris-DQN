import argparse
import torch
import cv2
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from game_env.tetris_game import Tetris
from rl_agent.models import DQN

def get_args():
    parser = argparse.ArgumentParser("Test a trained DQN agent for Tetris")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file")
    parser.add_argument("--width", type=int, default=10, help="Width of the Tetris grid")
    parser.add_argument("--height", type=int, default=20, help="Height of the Tetris grid")
    parser.add_argument("--block_size", type=int, default=30, help="Size of each block in pixels")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for rendering")
    parser.add_argument("--output", type=str, default="test_output.mp4", help="Output video file name")
    return parser.parse_args()

def test(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    model = DQN(input_shape=(3, opt.height, opt.width), num_actions=6).to(device)  # Assuming 6 actions
    model.load_state_dict(torch.load(opt.model_path, map_location=device))
    model.eval()

    # Initialize Tetris environment
    pygame.init()
    screen = pygame.display.set_mode((opt.width * opt.block_size, opt.height * opt.block_size))
    game = Tetris(screen)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 format
    out = cv2.VideoWriter(opt.output, fourcc, opt.fps, (opt.width * opt.block_size, opt.height * opt.block_size))

    # Test loop
    done = False
    state, _ = env.reset()

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # Select action using the model
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = model(state_tensor)
        action = torch.argmax(q_values).item()

        # Take the action
        next_state, reward, done, _, _ = env.step(action)

        # Render the game and write to video
        game.render(screen)
        pygame.display.flip()  # Update the display

        # Convert Pygame surface to OpenCV frame
        frame = pygame.surfarray.array3d(screen)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

        state = next_state

    out.release()
    pygame.quit()

if __name__ == "__main__":
    opt = get_args()
    test(opt)
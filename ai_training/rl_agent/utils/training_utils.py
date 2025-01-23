import torch

def soft_update(target_network, q_network, tau):
    """Updates the target network with a soft update using a factor of tau
    """
    for target_param, q_param in zip(target_network.parameters(), q_network.parameters()):
        target_param.data.copy_(tau * q_param.data + (1 - tau) * target_param.data)

def linear_epsilon_decay(epsilon_start, epsilon_end, num_episodes):
    """Creates a linear epsilon decay function.

    Args:
        epsilon_start (float): The starting epsilon value.
        epsilon_end (float): The ending epsilon value.
        num_episodes (int): The total number of training episodes.

    Returns:
        function: A function that takes the current episode and returns the decayed epsilon.
    """
    slope = (epsilon_end - epsilon_start) / num_episodes  # Calculate slope for linear decay
    def decay_function(episode):
        epsilon = max(epsilon_start + slope * episode, epsilon_end)
        return epsilon
    return decay_function
def tensor_to_numpy(tensor):
    """ Convert a pytorch tensor to numpy array """
    return tensor.detach().cpu().numpy()

def calculate_max_height(grid):
    """Calculate maximum column height in the grid"""
    heights = []
    for col in range(len(grid[0])):
        for row in range(len(grid)):
            if grid[row][col] != 0:
                heights.append(len(grid) - row)
                break
        else:
            heights.append(0)
    return max(heights)

def print_training_metrics(episode, avg_reward, epsilon, steps, total_reward,
                          lines_cleared, max_height, action_counts, buffer_size, 
                          total_episodes):
    print(f"\nEpisode: {episode}/{total_episodes}")
    print(f"  Reward: {total_reward:.1f} (Avg: {avg_reward:.2f}/step)")
    print(f"  Lines: {lines_cleared} | Max Height: {max_height}")
    print(f"  Epsilon: {epsilon:.3f} | Steps: {steps}")  # Added steps display
    print(f"  Actions: {action_counts}")
    print(f"  Buffer: {buffer_size} experiences")
    print("-" * 50)
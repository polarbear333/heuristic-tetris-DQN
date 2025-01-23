import os
class Logger:
    def __init__(self, config, filename="training_log.txt"):
        self.filename = os.path.join(config["model_path"], filename)
        self.file = open(self.filename, "w")
        # Add new headers
        self.file.write("Episode,Avg_Reward,Epsilon,Steps,Lines_Cleared,Max_Height,Action_Counts\n")

    def log(self, episode, avg_reward, epsilon, steps, lines_cleared, max_height, action_counts):
        # Format action counts as string
        action_str = "|".join([f"{k}:{v}" for k,v in action_counts.items()])
        log_line = (
            f"{episode},{avg_reward:.2f},{epsilon:.4f},"
            f"{steps},{lines_cleared},{max_height},\"{action_str}\"\n"
        )
        self.file.write(log_line)
        self.file.flush()

    def close(self):
        self.file.close()
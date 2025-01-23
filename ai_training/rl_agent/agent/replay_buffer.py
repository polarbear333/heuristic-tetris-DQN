import numpy as np
import torch
import random

class PrioritizedReplayBuffer:
    def __init__(self, buffer_size, alpha=0.6, beta=0.4, epsilon=1e-6):
        self.buffer_size = buffer_size
        self.buffer = []
        self.priorities = []
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.max_priority = 1.0


    def add(self, transition):
        self.buffer.append(transition)
        self.priorities.append(self.max_priority)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
            self.priorities.pop(0)

    def sample(self, batch_size):
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        sampled_transitions = [self.buffer[i] for i in indices]

        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        # Correctly unpack all 7 components
        (current_grids, current_features,
        actions, rewards,
        next_grids, next_features,
        dones) = zip(*sampled_transitions)

        return (
            torch.FloatTensor(np.array(current_grids)),
            torch.FloatTensor(np.array(current_features)),
            torch.LongTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)),
            torch.FloatTensor(np.array(next_grids)),
            torch.FloatTensor(np.array(next_features)),
            torch.BoolTensor(np.array(dones)),
            indices,
            weights
        )


    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon
        self.max_priority = max(self.priorities)
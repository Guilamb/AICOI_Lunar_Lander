import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import namedtuple

# Define experience replay buffer
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Define Q-network
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)  # Fully connected layer 1
        self.fc2 = nn.Linear(512, 256)  # Fully connected layer 2
        self.fc3 = nn.Linear(256, output_size)  # Output layer

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        return self.fc3(x)

# Initialize environment
env = gym.make('LunarLander-v2')
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

# Initialize Q-network and target network
q_network = QNetwork(input_size, output_size)
target_network = QNetwork(input_size, output_size)
target_network.load_state_dict(q_network.state_dict())

# Initialize replay buffer
replay_buffer = ReplayBuffer(capacity=10000)

# Define training parameters
num_episodes = 2000
batch_size = 32
gamma = 0.99
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 0.995

# Initialize optimizer
optimizer = optim.Adam(q_network.parameters(), lr=0.001)

# Epsilon-greedy exploration
def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()  # Random action
    else:
        with torch.no_grad():
            q_values = q_network(state)
            return q_values.argmax().item()  # Greedy action

# Preprocess state
def preprocess_state(state):
    return torch.tensor(state, dtype=torch.float32)

# Main training loop
epsilon = epsilon_start
for episode in range(num_episodes):
    state = env.reset()
    state = preprocess_state(state)
    total_reward = 0

    for t in range(10000):  # Maximum episode length
        action = select_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        next_state = preprocess_state(next_state)
        total_reward += reward

        replay_buffer.push(state, action, next_state, reward)
        state = next_state

        if len(replay_buffer) > batch_size:
            transitions = replay_buffer.sample(batch_size)
            batch = Transition(*zip(*transitions))

            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
            non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])

            state_batch = torch.stack(batch.state)
            action_batch = torch.tensor(batch.action)
            reward_batch = torch.tensor(batch.reward)
            next_state_values = torch.zeros(batch_size)
            next_state_values[non_final_mask] = target_network(non_final_next_states).max(1)[0].detach()

            expected_state_action_values = reward_batch + gamma * next_state_values

            # Compute Q-values
            state_action_values = q_network(state_batch).gather(1, action_batch.unsqueeze(1))

            # Compute loss
            loss = nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    epsilon = max(epsilon_final, epsilon * epsilon_decay)

    if episode % 100 == 0:
        target_network.load_state_dict(q_network.state_dict())
        print(f"Episode {episode}, Total Reward: {total_reward}")

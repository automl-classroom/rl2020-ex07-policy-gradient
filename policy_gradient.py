import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

MAX_EPISODE_LENGTH = 1000
DISCOUNT_FACTOR = 0.99
SEED = 0
MIN_BATCH_SIZE = 128

env = gym.make("CartPole-v1")
env.seed(SEED)
torch.manual_seed(SEED)


class Policy(nn.Module):
    """Define policy network"""

    def __init__(self, observation_space, action_space, hidden_size=128):
        super().__init__()

    def forward(self, x):
        pass


policy = Policy(env.observation_space, env.action_space)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)


def compute_returns(rewards, discount_factor=DISCOUNT_FACTOR):
    """Compute discounted returns"""



def policy_improvement(log_probs, rewards):
    """Compute REINFORCE policy gradient and perform gradient ascent step"""


def act(state):
    """ Use policy to sample an action and return probability for gradient update"""


def policy_gradient(num_episodes):
    rewards = []
    for episode in range(num_episodes):
        rewards.append(0)
        trajectory = []
        state = env.reset()

        for t in range(MAX_EPISODE_LENGTH):
            if episode % (num_episodes / 100) == 0:
                env.render()

            action, log_prob = act(state)

            next_state, reward, done, _ = env.step(action.item())

            trajectory.append((log_prob, reward))

            state = next_state
            rewards[-1] += reward

            if done:
                break

        loss = policy_improvement(*zip(*trajectory))

        if episode % (num_episodes / 100) == 0:
            print("Mean Reward: ", np.mean(rewards[-int(num_episodes / 100) :]))
    return rewards


if __name__ == "__main__":
    policy_gradient(1000)

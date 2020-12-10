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
        self.fc1 = nn.Linear(np.prod(observation_space.shape), hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_space.n)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        x = x.float()
        x = self.fc1(x)
        x = F.relu(x)
        logits = self.fc2(x)
        probs = F.softmax(logits, dim=-1)
        return probs


policy = Policy(env.observation_space, env.action_space)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)


def compute_returns(rewards, discount_factor=DISCOUNT_FACTOR):
    """Compute discounted returns"""
    returns = []
    return_to_go = 0
    for r in rewards[::-1]:
        return_to_go = r + discount_factor * return_to_go
        returns.insert(0, return_to_go)
    return returns


def policy_improvement(log_probs, rewards):
    """Compute REINFORCE policy gradient and perform gradient ascent step"""
    returns = torch.tensor(compute_returns(rewards))
    advantages = (returns - returns.mean()) / (
        torch.std(returns) + np.finfo(np.float32).eps
    )
    log_probs = torch.stack(log_probs)
    optimizer.zero_grad()
    loss = -torch.sum(log_probs * advantages)
    loss.backward()
    optimizer.step()
    return loss.item()


def act(state):
    """ Use policy to sample an action and return probability for gradient update"""
    probs = policy(state)
    distribution = Categorical(probs=probs)
    action = distribution.sample()
    log_prob = distribution.log_prob(action)
    return action, log_prob


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

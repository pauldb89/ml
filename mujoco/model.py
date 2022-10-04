from typing import Dict

import gym
import torch
from torch import nn
from torch.distributions import Distribution
from torch.distributions import MultivariateNormal


class GaussianMLPPolicy(nn.Module):
    def __init__(self, env: gym.Env, num_layers: int, hidden_dim: int):
        super().__init__()

        input_dim = env.observation_space.shape[0]
        output_dim = env.action_space.shape[0]
        layer_dims = [input_dim] + [hidden_dim] * num_layers + [output_dim]

        self.layers = nn.Sequential()
        for idx, (input_dim, output_dim) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            self.layers.append(nn.Linear(input_dim, output_dim))
            if idx < len(layer_dims) - 2:
                self.layers.append(nn.ReLU())

        self.log_std = nn.Parameter(torch.zeros(output_dim))

    def forward(self, states: torch.Tensor) -> Distribution:
        output = self.layers(states)
        return MultivariateNormal(loc=output, scale_tril=torch.diag(torch.exp(self.log_std)))

    def loss(self, states: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        distribution = self(states.cuda())
        return {
            "loss": -torch.mean(distribution.log_prob(actions.cuda())),
        }

    def sample(self, states: torch.Tensor) -> torch.Tensor:
        distribution = self(states.cuda())
        return distribution.sample()


class ExpertPolicyWrapper(nn.Module):
    def __init__(self, policy):
        super().__init__()

        self.policy = policy

    def loss(self, states: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {}

    def sample(self, states: torch.Tensor) -> torch.Tensor:
        return torch.squeeze(torch.from_numpy(self.policy.get_action(states.numpy())), dim=0)

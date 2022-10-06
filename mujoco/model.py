from typing import Dict

import gym
import torch
from torch import nn
import torch.nn.functional as F

from torch.distributions import Categorical
from torch.distributions import Distribution
from torch.distributions import MultivariateNormal


class BasePolicy(nn.Module):
    def imitation_loss(self, states: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        distribution = self(states.cuda())
        return {
            "loss": -torch.mean(distribution.log_prob(actions.cuda())),
        }

    def policy_gradient_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        distribution = self(states.cuda())
        return {
            "loss": -torch.mean(distribution.log_prob(actions.cuda()) * returns.cuda())
        }

    def sample(self, states: torch.Tensor) -> torch.Tensor:
        distribution = self(states.cuda())
        return distribution.sample()


class GaussianMLPPolicy(BasePolicy):
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


class ExpertPolicyWrapper(nn.Module):
    def __init__(self, policy):
        super().__init__()

        self.policy = policy

    def imitation_loss(self, states: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {}

    def sample(self, states: torch.Tensor) -> torch.Tensor:
        return torch.squeeze(torch.from_numpy(self.policy.get_action(states.numpy())), dim=0)


class DiscreteMLPPolicy(BasePolicy):
    def __init__(self, env: gym.Env, num_layers: int, hidden_dim: int):
        super().__init__()

        input_dim = env.observation_space.shape[0]
        output_dim = env.action_space.n
        layer_dims = [input_dim] + [hidden_dim] * num_layers + [output_dim]

        self.layers = nn.Sequential()
        for idx, (input_dim, output_dim) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            self.layers.append(nn.Linear(input_dim, output_dim))
            if idx < len(layer_dims) - 2:
                self.layers.append(nn.ReLU())

    def forward(self, states: torch.Tensor) -> Distribution:
        logits = self.layers(states.cuda())
        return Categorical(logits=logits)


class BaselineEstimator(nn.Module):
    def __init__(self, env: gym.Env, num_layers: int, hidden_dim: int):
        super().__init__()

        input_dim = env.observation_space.shape[0]
        layer_dims = [input_dim] + [hidden_dim] * num_layers + [1]

        self.layers = nn.Sequential()
        for idx, (input_dim, output_dim) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            self.layers.append(nn.Linear(input_dim, output_dim))
            if idx < len(layer_dims) - 2:
                self.layers.append(nn.ReLU())

    def forward(self, states: torch.Tensor, target_rewards: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(torch.squeeze(self.layers(states.cuda()), dim=1), target_rewards.cuda())

    def eval_forward(self, states: torch.Tensor) -> torch.Tensor:
        return torch.squeeze(self.layers(states.cuda()), dim=1)

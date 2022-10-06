import gym
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer

from common.solvers.policy_gradient import Batch
from common.solvers.policy_gradient import PolicySolver


class ActorCriticSolver(PolicySolver):
    def __init__(
        self,
        train_env: gym.Env,
        eval_env: gym.Env,
        policy: DistributedDataParallel,
        baseline_estimator: DistributedDataParallel,
        policy_optimizer: Optimizer,
        baseline_optimizer: Optimizer,
        epochs: int,
        num_samples_per_epoch: int,
        discount: float,
        baseline_train_steps_per_epoch: int,
        baseline_batch_size: int,
        normalize_advantage: bool,
    ):
        super().__init__(
            train_env=train_env,
            eval_env=eval_env,
            policy=policy,
            epochs=epochs,
            num_samples_per_epoch=num_samples_per_epoch,
            discount=discount,
        )

        self.baseline_estimator = baseline_estimator
        self.policy_optimizer = policy_optimizer
        self.baseline_optimizer = baseline_optimizer

        self.baseline_train_steps_per_epoch = baseline_train_steps_per_epoch
        self.baseline_batch_size = baseline_batch_size
        self.normalize_advantage = normalize_advantage

    def train_step(self, epoch: int, batch: Batch):
        baseline_losses = []
        for _ in range(self.baseline_train_steps_per_epoch):
            # TODO(pauldb): Try using bootstrapped targets instead of Monte Carlo targets.
            indices = torch.randperm(batch.returns.size(0))[:self.baseline_batch_size]
            loss = self.baseline_estimator(states=batch.states[indices], target_rewards=batch.returns[indices])
            baseline_losses.append(loss.item())

            self.baseline_optimizer.zero_grad()
            loss.backward()
            self.baseline_optimizer.step()

        with torch.inference_mode():
            self.baseline_estimator.eval()
            state_values = self.baseline_estimator.module.eval_forward(batch.states)
            self.baseline_estimator.train()

        next_state_values = torch.where(
            batch.terminal_states.cuda(),
            torch.zeros_like(state_values),
            F.pad(state_values[1:], (0, 1)),
        )
        advantages = batch.rewards.cuda() + self.discount * next_state_values - state_values
        if self.normalize_advantage:
            # TODO(pauldb): Should the advantages be normalized also only at timestep t?
            advantages = (advantages - torch.mean(advantages)) / (torch.std(advantages) + 1e-5)

        loss_metrics = self.policy.module.policy_gradient_loss(
            states=batch.states,
            actions=batch.actions,
            returns=advantages,
        )

        self.policy_optimizer.zero_grad()
        loss_metrics["loss"].backward()
        self.policy_optimizer.step()

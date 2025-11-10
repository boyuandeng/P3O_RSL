# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.utils import split_and_pad_trajectories


class RolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.actions = None
            self.privileged_actions = None
            self.rewards = None
            self.dones = None
            self.values_r = None      # [N,1]
            # cost values
            self.values_c = None      # [N,m]
            # failure probability values
            self.values_fail = None      # [N,1]

            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None
            # cost signals
            self.costs = None         # [N,m]
            # failure probability signals
            self.costs_fail = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        obs,
        actions_shape,
        device="cpu",
        num_costs: int = 0,
        predict_failure_prob: bool = False,
    ):
        # store inputs
        self.training_type = training_type
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.actions_shape = actions_shape
        self.num_costs = int(num_costs)
        self.predict_failure_prob = predict_failure_prob
        # Core
        self.observations = TensorDict(
            {key: torch.zeros(num_transitions_per_env, *value.shape, device=device) for key, value in obs.items()},
            batch_size=[num_transitions_per_env, num_envs],
            device=self.device,
        )
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # for distillation
        if training_type == "distillation":
            self.privileged_actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        # for reinforcement learning
        if training_type == "rl":
            self.values_r = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
            self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
            self.returns_r  = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.advantages_r  = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            #for cost values
            if self.num_costs > 0:
                self.values_c = torch.zeros(num_transitions_per_env, num_envs, self.num_costs, device=device)
                self.costs = torch.zeros(num_transitions_per_env, num_envs, self.num_costs, device=device)
                self.returns_c = torch.zeros(num_transitions_per_env, num_envs, self.num_costs, device=device)
                self.advantages_c = torch.zeros(num_transitions_per_env, num_envs, self.num_costs, device=device)
            #for failure probability values
            if self.predict_failure_prob:
                self.values_fail = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
                self.costs_fail = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
                self.returns_fail = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
                self.advantages_fail = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        # For RNN networks
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        # counter for the number of transitions stored
        self.step = 0

    def add_transitions(self, transition: Transition):
        # check if the transition is valid
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")

        # Core
        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))

        # for distillation
        if self.training_type == "distillation":
            self.privileged_actions[self.step].copy_(transition.privileged_actions)

        # for reinforcement learning
        if self.training_type == "rl":
            self.values_r[self.step].copy_(transition.values_r)
            self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
            self.mu[self.step].copy_(transition.action_mean)
            self.sigma[self.step].copy_(transition.action_sigma)
            if self.num_costs > 0 and transition.values_c is not None:
                self.values_c[self.step].copy_(transition.values_c)
                self.costs[self.step].copy_(transition.costs)
            if self.predict_failure_prob and transition.values_fail is not None:
                self.values_fail[self.step].copy_(transition.values_fail)
                self.costs_fail[self.step].copy_(transition.costs_fail)

        # For RNN networks
        self._save_hidden_states(transition.hidden_states)

        # increment the counter
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states == (None, None):
            return
        # make a tuple out of GRU hidden state sto match the LSTM format
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)
        # initialize if needed
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [
                torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))
            ]
            self.saved_hidden_states_c = [
                torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))
            ]
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values_r, gamma, lam, normalize_advantage: bool = True,
        last_values_c=None,
        gamma_cost=None,
        lam_cost=None,
        last_values_fail=None,
        gamma_cost_fail=None,
        lam_cost_fail=None,):
        advantage_r  = 0
        for step in reversed(range(self.num_transitions_per_env)):
            # if we are at the last step, bootstrap the return value
            if step == self.num_transitions_per_env - 1:
                next_values = last_values_r
            else:
                next_values = self.values_r[step + 1]
            # 1 if we are not in a terminal state, 0 otherwise
            next_is_not_terminal = 1.0 - self.dones[step].float()
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values_r[step]
            # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
            advantage_r  = delta + next_is_not_terminal * gamma * lam * advantage_r
            # Return: R_t = A(s_t, a_t) + V(s_t)
            self.returns_r[step] = advantage_r + self.values_r[step]

        # Compute the advantages
        self.advantages_r = self.returns_r - self.values_r
        # Normalize the advantages if flag is set
        # This is to prevent double normalization (i.e. if per minibatch normalization is used)
        if normalize_advantage:
            self.advantages_r  = (self.advantages_r  - self.advantages_r .mean()) / (self.advantages_r .std() + 1e-8)
        if last_values_c is not None:
            gc = gamma_cost if gamma_cost is not None else gamma
            lc = lam_cost if lam_cost is not None else lam
            advantage_c = torch.zeros_like(self.costs[0])  # [N,m]
            # assert self.costs.shape    == (self.num_transitions_per_env, self.num_envs, self.num_costs)
            # assert self.values_c.shape == (self.num_transitions_per_env, self.num_envs, self.num_costs)
            # assert last_values_c.shape == (self.num_envs, self.num_costs)
            for step in reversed(range(self.num_transitions_per_env)):
                next_vc = last_values_c if step == self.num_transitions_per_env - 1 else self.values_c[step + 1]
                next_nonterm = (1.0 - self.dones[step].float())
                # assert next_is_not_terminal.shape == (self.num_envs, 1), f"got {next_is_not_terminal.shape}"
                delta_c = self.costs[step] + next_nonterm * gc * next_vc - self.values_c[step]
                advantage_c = delta_c + next_nonterm * gc * lc * advantage_c
                self.returns_c[step] = advantage_c + self.values_c[step]
            self.advantages_c = self.returns_c - self.values_c
            if normalize_advantage:
                mean = self.advantages_c.mean(dim=(0, 1), keepdim=True)
                std = self.advantages_c.std(dim=(0, 1), keepdim=True)
                self.advantages_c = (self.advantages_c - mean) / (std + 1e-8)
        if last_values_fail is not None and self.predict_failure_prob:
            gc_f = gamma_cost_fail if gamma_cost_fail is not None else gamma
            lc_f = lam_cost_fail if lam_cost_fail is not None else lam
            advantage_f = torch.zeros_like(self.costs_fail[0])  # [N, 1]
            
            for step in reversed(range(self.num_transitions_per_env)):
                next_vf = last_values_fail if step == self.num_transitions_per_env - 1 else self.values_fail[step + 1]
                next_nonterm = (1.0 - self.dones[step].float())
                
                delta_f = self.costs_fail[step] + next_nonterm * gc_f * next_vf - self.values_fail[step]
                advantage_f = delta_f + next_nonterm * gc_f * lc_f * advantage_f
                self.returns_fail[step] = advantage_f + self.values_fail[step]

            self.advantages_fail = self.returns_fail - self.values_fail
    # for distillation
    def generator(self):
        if self.training_type != "distillation":
            raise ValueError("This function is only available for distillation training.")

        for i in range(self.num_transitions_per_env):
            yield self.observations[i], self.actions[i], self.privileged_actions[i], self.dones[i]

    # for reinforcement learning with feedforward networks
    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        # Core
        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values_r = self.values_r.flatten(0, 1)
        returns_r = self.returns_r.flatten(0, 1)
        advantages_r = self.advantages_r.flatten(0, 1)
        # For PPO
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)
        if self.num_costs > 0:
            values_c = self.values_c.flatten(0, 1)
            returns_c = self.returns_c.flatten(0, 1)
            advantages_c = self.advantages_c.flatten(0, 1)
        if self.predict_failure_prob:
            values_fail = self.values_fail.flatten(0, 1)
            returns_fail = self.returns_fail.flatten(0, 1)
            advantages_fail = self.advantages_fail.flatten(0, 1)
        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                # Select the indices for the mini-batch
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                # Create the mini-batch
                # -- Core
                obs_batch = observations[batch_idx]
                actions_batch = actions[batch_idx]

                # -- For PPO
                target_values_batch = values_r[batch_idx]
                returns_r_batch = returns_r[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_r_batch = advantages_r[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                if self.num_costs > 0:
                    target_values_c_batch = values_c[batch_idx]
                    returns_c_batch = returns_c[batch_idx]
                    advantages_c_batch = advantages_c[batch_idx]
                else:
                    target_values_c_batch = None
                    returns_c_batch = None
                    advantages_c_batch = None
                if self.predict_failure_prob:
                    target_values_fail_batch = values_fail[batch_idx]
                    returns_fail_batch = returns_fail[batch_idx]
                    advantages_fail_batch = advantages_fail[batch_idx]
                else:
                    target_values_fail_batch = None
                    returns_fail_batch = None
                    advantages_fail_batch = None
                # yield the mini-batch
                yield obs_batch, actions_batch, target_values_batch, advantages_r_batch, returns_r_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
                    None,
                    None,
                ), None,target_values_c_batch, advantages_c_batch, returns_c_batch,\
                    target_values_fail_batch, advantages_fail_batch, returns_fail_batch

    # for reinfrocement learning with recurrent networks
    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

                # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
                # then take only time steps after dones (flattens num envs and time dimensions),
                # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
                last_was_done = last_was_done.permute(1, 0)
                hid_a_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_a
                ]
                hid_c_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_c
                ]
                # remove the tuple for GRU
                hid_a_batch = hid_a_batch[0] if len(hid_a_batch) == 1 else hid_a_batch
                hid_c_batch = hid_c_batch[0] if len(hid_c_batch) == 1 else hid_c_batch

                yield obs_batch, actions_batch, values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
                    hid_a_batch,
                    hid_c_batch,
                ), masks_batch

                first_traj = last_traj

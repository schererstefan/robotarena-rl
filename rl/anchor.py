"""KL anchor to a frozen reference policy (Experiment 1, Sep 2026).

AnchorKLPPO subclasses SB3 PPO and adds a soft KL penalty pulling the live
policy toward a frozen anchor (the iteration's seed checkpoint, i.e. the
global best so far):

    loss += anchor_coef(t) * mean_s KL( pi_live(.|s) || pi_anchor(.|s) )

``anchor_coef(t)`` starts at ``kl_anchor_coef`` and anneals linearly to 0
over the run when ``kl_anchor_anneal`` is set, so the anchor resists
catastrophic forgetting of the fixed-bot skills early while never
permanently capping exploration.

The anchor policy was trained under its own VecNormalize stats, so live
(normalized) observations are mapped back through the *live* stats and
re-normalized with the *anchor* stats before the anchor forward pass:

    raw       = obs_live * sqrt(var_live) + mean_live
    obs_anch  = (raw - mean_anchor) / sqrt(var_anchor)

Written against stable_baselines3==2.9.0: train() is copied verbatim from
that version with the anchor term added. Re-verify against the installed
SB3 source before upgrading the dependency.
"""
from __future__ import annotations

import pickle

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance
from torch.nn import functional as F


class AnchorKLPPO(PPO):
    def __init__(self, *args,
                 anchor_model_path: str | None = None,
                 anchor_vecnormalize_path: str | None = None,
                 kl_anchor_coef: float = 0.0,
                 kl_anchor_anneal: bool = True,
                 **kwargs):
        # NOTE: SB3's BaseAlgorithm.load() does NOT forward kwargs to the
        # constructor -- it builds with defaults, then does
        # ``model.__dict__.update(kwargs)`` and calls ``_setup_model()``.
        # So the anchor is (re)loaded in _setup_model(), never here.
        self.anchor_model_path = anchor_model_path
        self.anchor_vecnormalize_path = anchor_vecnormalize_path
        self.kl_anchor_coef = float(kl_anchor_coef)
        self.kl_anchor_anneal = bool(kl_anchor_anneal)
        self.anchor_policy = None
        self._anchor_mean: th.Tensor | None = None
        self._anchor_var: th.Tensor | None = None
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        # (Re)load the frozen anchor. Always reload from the configured path
        # so a model.zip saved by a previous AnchorKLPPO run can never leak
        # a stale anchor policy through a pickle round-trip.
        self.anchor_policy = None
        self._anchor_mean = self._anchor_var = None
        if self.anchor_model_path:
            anchor = PPO.load(self.anchor_model_path)
            self.anchor_policy = anchor.policy
            self.anchor_policy.to(self.device)
            self.anchor_policy.set_training_mode(False)
            for p in self.anchor_policy.parameters():
                p.requires_grad_(False)
        if self.anchor_vecnormalize_path:
            with open(self.anchor_vecnormalize_path, "rb") as f:
                vn = pickle.load(f)
            self._anchor_mean = th.as_tensor(
                np.asarray(vn.obs_rms.mean, dtype=np.float32),
                device=self.device)
            self._anchor_var = th.as_tensor(
                np.asarray(vn.obs_rms.var, dtype=np.float32),
                device=self.device)

    def _excluded_save_params(self) -> list[str]:
        # Never pickle the frozen anchor into checkpoints (it would double
        # their size); it is reloaded from anchor_model_path on load.
        return super()._excluded_save_params() + [
            "anchor_policy", "_anchor_mean", "_anchor_var",
        ]

    def _anchor_kl(self, obs: th.Tensor) -> th.Tensor:
        """mean_s KL(pi_live(.|s) || pi_anchor(.|s)) over a minibatch."""
        assert self.anchor_policy is not None
        vec_norm = self.get_vec_normalize_env()
        if vec_norm is not None and self._anchor_mean is not None:
            live_mean = th.as_tensor(
                np.asarray(vec_norm.obs_rms.mean, dtype=np.float32),
                device=self.device)
            live_var = th.as_tensor(
                np.asarray(vec_norm.obs_rms.var, dtype=np.float32),
                device=self.device)
            raw = obs * th.sqrt(live_var + 1e-8) + live_mean
            obs_anchor = ((raw - self._anchor_mean)
                          / th.sqrt(self._anchor_var + 1e-8))
        else:
            # No VecNormalize stats available: approximate with live obs.
            obs_anchor = obs
        with th.no_grad():
            anchor_dist = self.anchor_policy.get_distribution(obs_anchor)
        live_dist = self.policy.get_distribution(obs)
        # MultiDiscrete -> list of Categoricals; Discrete -> a single one.
        live_ds = (list(live_dist.distribution)
                   if isinstance(live_dist.distribution, (list, tuple))
                   else [live_dist.distribution])
        anch_ds = (list(anchor_dist.distribution)
                   if isinstance(anchor_dist.distribution, (list, tuple))
                   else [anchor_dist.distribution])
        kl = th.zeros((), device=self.device)
        for dl, da in zip(live_ds, anch_ds):
            p = dl.probs
            q = da.probs.clamp_min(1e-8)
            kl = kl + (p * (p.clamp_min(1e-8).log() - q.log())).sum(-1)
        return kl.mean()

    # ------------------------------------------------------------------
    # PPO.train from stable_baselines3==2.9.0, verbatim, plus the anchor
    # KL term (marked ANCHOR below). Keep in sync if SB3 is upgraded.
    # ------------------------------------------------------------------
    def train(self) -> None:
        """Update policy using the currently gathered rollout buffer."""
        if self.anchor_policy is None or self.kl_anchor_coef <= 0:
            super().train()
            return
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]
        # ANCHOR: annealed KL coefficient toward the frozen anchor policy.
        if self.kl_anchor_anneal:
            anchor_coef = self.kl_anchor_coef * self._current_progress_remaining
        else:
            anchor_coef = self.kl_anchor_coef

        entropy_losses = []
        pg_losses, value_losses = [], []
        anchor_kls = []  # ANCHOR
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                # ANCHOR: KL(pi_live || pi_anchor), annealed coefficient.
                anchor_kl = self._anchor_kl(rollout_data.observations)
                anchor_kls.append(anchor_kl.item())

                loss = (policy_loss + self.ent_coef * entropy_loss
                        + self.vf_coef * value_loss + anchor_coef * anchor_kl)

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/anchor_kl", np.mean(anchor_kls))  # ANCHOR
        self.logger.record("train/anchor_coef", anchor_coef)  # ANCHOR
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

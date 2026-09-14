"""The PPO update: clipped surrogate + value loss + entropy bonus.

Together with `rl/policy.py` this is the only place torch appears. Trajectories
arrive as plain Python from `rl/rollout.py` and become tensors here.

The masks stored on each `DecisionRecord` are reused verbatim at update time.
That is not an optimisation: recomputing or omitting them would measure the
ratio pi_new/pi_old against a different distribution than the one that actually
acted, which silently corrupts the gradient.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from pokerlab.rl.action_space import ACTION_DIM
from pokerlab.rl.features import FEATURE_VERSION, OBS_DIM
from pokerlab.rl.policy import PokerActorCritic
from pokerlab.rl.rollout import HandTrajectory


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 3e-4
    clip_epsilon: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    epochs: int = 4
    minibatch_size: int = 1024
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True


@dataclass
class TrainingBatch:
    features: Tensor  # (N, OBS_DIM)
    masks: Tensor  # (N, ACTION_DIM) bool
    actions: Tensor  # (N,) int64
    old_log_probs: Tensor  # (N,)
    advantages: Tensor  # (N,)
    returns: Tensor  # (N,)

    def __len__(self) -> int:
        return int(self.features.shape[0])


def build_batch(
    trajectories: list[HandTrajectory], *, device: str | torch.device = "cpu"
) -> TrainingBatch:
    """Flatten trajectories into aligned tensors.

    Decisions, advantages and returns are zipped together rather than
    accumulated separately, so the three can never fall out of step.
    """
    features: list[list[float]] = []
    masks: list[list[bool]] = []
    actions: list[int] = []
    old_log_probs: list[float] = []
    advantages: list[float] = []
    returns: list[float] = []

    for trajectory in trajectories:
        for decision, advantage, value_target in zip(
            trajectory.decisions, trajectory.advantages, trajectory.returns
        ):
            features.append(decision.features)
            masks.append(decision.legal_mask)
            actions.append(decision.action_index)
            old_log_probs.append(decision.log_prob)
            advantages.append(advantage)
            returns.append(value_target)

    return TrainingBatch(
        features=torch.tensor(features, dtype=torch.float32, device=device),
        masks=torch.tensor(masks, dtype=torch.bool, device=device),
        actions=torch.tensor(actions, dtype=torch.int64, device=device),
        old_log_probs=torch.tensor(old_log_probs, dtype=torch.float32, device=device),
        advantages=torch.tensor(advantages, dtype=torch.float32, device=device),
        returns=torch.tensor(returns, dtype=torch.float32, device=device),
    )


def ppo_update(
    model: PokerActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: TrainingBatch,
    config: PPOConfig,
) -> dict[str, float]:
    """One PPO update over a batch. Returns diagnostics, averaged per minibatch."""
    advantages = batch.advantages
    if config.normalize_advantages and len(batch) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0,
              "clip_fraction": 0.0, "grad_norm": 0.0}
    steps = 0

    for _ in range(config.epochs):
        permutation = torch.randperm(len(batch), device=batch.features.device)
        for start in range(0, len(batch), config.minibatch_size):
            index = permutation[start : start + config.minibatch_size]
            logits, values = model(batch.features[index], batch.masks[index])
            distribution = torch.distributions.Categorical(logits=logits)

            log_probs = distribution.log_prob(batch.actions[index])
            log_ratio = log_probs - batch.old_log_probs[index]
            ratio = log_ratio.exp()
            minibatch_advantages = advantages[index]

            unclipped = ratio * minibatch_advantages
            clipped = ratio.clamp(1 - config.clip_epsilon, 1 + config.clip_epsilon)
            policy_loss = -torch.min(unclipped, clipped * minibatch_advantages).mean()
            value_loss = nn.functional.mse_loss(values, batch.returns[index])
            entropy = distribution.entropy().mean()

            loss = (
                policy_loss
                + config.value_coefficient * value_loss
                - config.entropy_coefficient * entropy
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                totals["policy_loss"] += float(policy_loss)
                totals["value_loss"] += float(value_loss)
                totals["entropy"] += float(entropy)
                # Schulman's low-variance KL estimator; stays non-negative.
                totals["approx_kl"] += float(((ratio - 1) - log_ratio).mean())
                totals["clip_fraction"] += float(
                    ((ratio - 1).abs() > config.clip_epsilon).float().mean()
                )
                totals["grad_norm"] += float(grad_norm)
            steps += 1

    return {name: value / max(steps, 1) for name, value in totals.items()}


def save_checkpoint(
    path: str | Path,
    model: PokerActorCritic,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    iteration: int = 0,
    metadata: dict[str, Any] | None = None,
) -> None:
    """The encoding fingerprint and the network shape travel with the weights,
    so a stale or differently-sized checkpoint fails loudly on load instead of
    silently mismatching, and an archived model can be rebuilt without the
    caller remembering how it was configured."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "obs_dim": OBS_DIM,
            "action_dim": ACTION_DIM,
            "feature_version": FEATURE_VERSION,
            "hidden": model.hidden,
            "num_layers": model.num_layers,
            "iteration": iteration,
            "model": model.state_dict(),
            "optimizer": None if optimizer is None else optimizer.state_dict(),
            "metadata": metadata or {},
        },
        path,
    )


class IncompatibleCheckpointError(ValueError):
    """The checkpoint was trained against a different observation encoding."""


def check_compatible(checkpoint: dict[str, Any], source: str = "checkpoint") -> None:
    if checkpoint.get("obs_dim") != OBS_DIM or checkpoint.get("action_dim") != ACTION_DIM:
        raise IncompatibleCheckpointError(
            f"{source} was trained on obs_dim={checkpoint.get('obs_dim')} "
            f"action_dim={checkpoint.get('action_dim')}, but this build uses "
            f"{OBS_DIM}/{ACTION_DIM} -- the encoding changed since it was saved"
        )
    saved_version = checkpoint.get("feature_version", 0)
    if saved_version != FEATURE_VERSION:
        raise IncompatibleCheckpointError(
            f"{source} was trained on feature version {saved_version}, this build is "
            f"{FEATURE_VERSION} -- the features have the same shape but a different "
            f"meaning, so the weights would read the wrong thing from every slot"
        )


def load_checkpoint(
    path: str | Path,
    model: PokerActorCritic,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    check_compatible(checkpoint, str(path))
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and checkpoint["optimizer"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def build_model_from_checkpoint(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> tuple[PokerActorCritic, dict[str, Any]]:
    """Rebuild a saved network at the shape it was trained with, frozen for
    inference. Used to seat previously trained agents as opponents."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    check_compatible(checkpoint, str(path))
    model = PokerActorCritic(
        hidden=checkpoint.get("hidden", 512), num_layers=checkpoint.get("num_layers", 3)
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint

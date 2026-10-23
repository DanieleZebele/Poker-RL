"""The PPO update: clipped surrogate + value loss + entropy bonus.

Together with `rl/policy.py` this is the only place torch appears. Trajectories
arrive as plain Python from `rl/rollout.py` and become tensors here.

The masks stored on each `DecisionRecord` are reused verbatim at update time.
That is not an optimisation: recomputing or omitting them would measure the
ratio pi_new/pi_old against a different distribution than the one that actually
acted, which silently corrupts the gradient.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from pokerlab.rl.action_space import ACTION_DIM
from pokerlab.rl.features import SUPPORTED_VERSIONS, observation_dim
from pokerlab.rl.policy import SHAPE_KEYS, PokerActorCritic
from pokerlab.rl.rollout import HandTrajectory


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 3e-4
    clip_epsilon: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.0
    # The entropy bonus falls from `entropy_coefficient` to this over `entropy_anneal_iterations`
    # iterations and stays there (see `entropy_coefficient_at`); 0 iterations = constant.
    entropy_coefficient_final: float = 0.0
    entropy_anneal_iterations: int = 0
    # The learning rate falls geometrically from `learning_rate` to `learning_rate_final`
    # over `learning_rate_anneal_iterations` iterations and stays there (see
    # `learning_rate_at`); 0 iterations = constant.
    learning_rate_anneal_iterations: int = 0
    learning_rate_final: float = 1e-5
    epochs: int = 4
    minibatch_size: int = 1024
    # The gradient-norm clip, one per network: the policy and the critic share no weight,
    # and their gradients live on different scales (the policy's loss is of order 1 on
    # normalised advantages, the critic's is a squared error in the reward's unit), so one
    # clip over both would cut each by a factor the other decides.
    policy_max_grad_norm: float = 0.5
    critic_max_grad_norm: float = 1.0
    normalize_advantages: bool = True
    # Each decision's squared error in the critic's loss is weighed by its chips at stake,
    # in big blinds, to the power -critic_stack_power (0: every decision alike). The
    # spread of the value target grows with the stake, so an unweighted loss is spent on
    # the deep hands and the short ones are left out: inverse-variance weighting.
    # A weight that is a function of the state alone leaves the minimiser where it was,
    # E[G | s], for every state -- only the critic's capacity moves between them -- and
    # any state-only critic is still an unbiased baseline, so the policy's gradient
    # keeps its objective. Never weigh the policy's loss this way: that would change it.
    critic_stack_power: float = 0.0

    def __post_init__(self) -> None:
        if self.learning_rate_anneal_iterations < 0:
            raise ValueError("learning_rate_anneal_iterations cannot be negative")
        if self.learning_rate_anneal_iterations and (
            self.learning_rate <= 0 or self.learning_rate_final <= 0
        ):
            raise ValueError(
                "annealing the learning rate needs a positive start and a positive final value: "
                "it falls geometrically, and a geometric curve never leaves or reaches 0"
            )
        if self.entropy_anneal_iterations < 0:
            raise ValueError("entropy_anneal_iterations cannot be negative")
        if self.entropy_anneal_iterations and (
            self.entropy_coefficient <= 0 or self.entropy_coefficient_final <= 0
        ):
            raise ValueError(
                "annealing the entropy bonus needs a positive start and a positive final value: "
                "it falls geometrically, and a geometric curve never leaves or reaches 0"
            )


def entropy_coefficient_at(config: PPOConfig, iteration: int) -> float:
    """The entropy bonus of a run's `iteration`-th update (1-based).

    Geometric from `entropy_coefficient` at the first iteration to
    `entropy_coefficient_final` after `entropy_anneal_iterations` more, then flat: the
    bonus spans orders of magnitude, so a straight line would spend nearly all the time
    near the start value. The iteration is the run's own, so every worker follows the
    same curve whatever it inherited: a child that resumed a trained parent starts its
    run at the high end again."""
    steps = config.entropy_anneal_iterations
    if steps == 0:
        return config.entropy_coefficient
    progress = min(max(iteration - 1, 0) / steps, 1.0)
    return config.entropy_coefficient * (config.entropy_coefficient_final / config.entropy_coefficient) ** progress


def learning_rate_at(config: PPOConfig, iteration: int) -> float:
    """The learning rate of a run's `iteration`-th update (1-based).

    Geometric from `learning_rate` (the worker's own, swept, value) at the first iteration
    to the fleet's `learning_rate_final` (an absolute value) after
    `learning_rate_anneal_iterations` more, then flat. Counted from the start of the run, like the entropy bonus: a child that
    resumed a trained parent starts at its own `lr` again."""
    steps = config.learning_rate_anneal_iterations
    if steps == 0:
        return config.learning_rate
    progress = min(max(iteration - 1, 0) / steps, 1.0)
    return config.learning_rate * (config.learning_rate_final / config.learning_rate) ** progress


@dataclass
class TrainingBatch:
    features: Tensor  # (N, OBS_DIM)
    masks: Tensor  # (N, ACTION_DIM) bool
    actions: Tensor  # (N,) int64
    old_log_probs: Tensor  # (N,)
    advantages: Tensor  # (N,)
    returns: Tensor  # (N,)
    # What the critic reads (`rl/policy.py`): the equity of every seat of the decision's hand.
    critic_extra: Tensor  # (N, EQUITY_SLOTS)
    # Each decision's chips at stake in big blinds (`DecisionRecord.stake_bb`).
    stakes: Tensor  # (N,)

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
    critic_extra: list[list[float]] = []
    stakes: list[float] = []

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
            if decision.critic_extra is None:
                raise ValueError("a decision reached the batch without what the critic reads")
            critic_extra.append(decision.critic_extra)
            stakes.append(decision.stake_bb)

    return TrainingBatch(
        features=torch.tensor(features, dtype=torch.float32, device=device),
        masks=torch.tensor(masks, dtype=torch.bool, device=device),
        actions=torch.tensor(actions, dtype=torch.int64, device=device),
        old_log_probs=torch.tensor(old_log_probs, dtype=torch.float32, device=device),
        advantages=torch.tensor(advantages, dtype=torch.float32, device=device),
        returns=torch.tensor(returns, dtype=torch.float32, device=device),
        critic_extra=torch.tensor(critic_extra, dtype=torch.float32, device=device),
        stakes=torch.tensor(stakes, dtype=torch.float32, device=device),
    )


# The stake below which a decision is weighed as if it were this deep: a seat with less
# than a big blind behind is rare, and its weight would otherwise grow without bound.
MIN_STAKE_BB = 1.0


def critic_weights(stakes: Tensor, power: float) -> Tensor:
    """The weight of each decision in the critic's loss: stake ** -power, rescaled to a
    mean of 1 over the batch so the loss keeps its scale (and the critic's gradient clip
    its meaning) whatever the power."""
    if power == 0.0:
        return torch.ones_like(stakes)
    weights = stakes.clamp(min=MIN_STAKE_BB).pow(-power)
    return weights / weights.mean()


def ppo_update(
    model: PokerActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: TrainingBatch,
    config: PPOConfig,
) -> dict[str, float]:
    """One PPO update over a batch. Returns diagnostics, averaged per minibatch.

    Each network's gradient is clipped to its own threshold, and reported per network as
    the mean norm before the cut, its sd over the steps and the share of steps cut
    (`grad_norm_policy`, `grad_norm_policy_sd`, `grad_clipped_policy` and the same for the
    critic, see `rl/grad_log.py`): the mean alone does not say how often the cut fires."""
    advantages = batch.advantages
    # Raw statistics, read before normalising: after it they are 0 and 1 by
    # construction, which says nothing about the scale the GAE produced.
    raw_mean = float(advantages.mean())
    raw_std = float(advantages.std()) if len(batch) > 1 else 0.0
    if config.normalize_advantages and len(batch) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    value_weights = critic_weights(batch.stakes, config.critic_stack_power)

    totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0,
              "clip_fraction": 0.0}
    steps = 0
    # The norm of every step, before the cut (what `clip_grad_norm_` returns), per network.
    policy_norms: list[float] = []
    critic_norms: list[float] = []

    # Every minibatch has exactly `minibatch_size` decisions: the remainder that would make a
    # smaller last one is left out of this epoch (the next one shuffles again, so it is a
    # different handful each time). A batch smaller than one minibatch is used whole.
    full = len(batch) - len(batch) % config.minibatch_size
    stop = full if full else len(batch)

    for _ in range(config.epochs):
        permutation = torch.randperm(len(batch), device=batch.features.device)
        for start in range(0, stop, config.minibatch_size):
            index = permutation[start : start + config.minibatch_size]
            logits, values = model(
                batch.features[index],
                batch.masks[index],
                batch.critic_extra[index],
            )
            distribution = torch.distributions.Categorical(logits=logits)

            log_probs = distribution.log_prob(batch.actions[index])
            log_ratio = log_probs - batch.old_log_probs[index]
            ratio = log_ratio.exp()
            minibatch_advantages = advantages[index]

            unclipped = ratio * minibatch_advantages
            clipped = ratio.clamp(1 - config.clip_epsilon, 1 + config.clip_epsilon)
            policy_loss = -torch.min(unclipped, clipped * minibatch_advantages).mean()
            squared_errors = (values - batch.returns[index]) ** 2
            value_loss = (value_weights[index] * squared_errors).mean()
            entropy = distribution.entropy().mean()

            loss = (
                policy_loss
                + config.value_coefficient * value_loss
                - config.entropy_coefficient * entropy
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            policy_norms.append(float(
                nn.utils.clip_grad_norm_(model.policy_parameters(), config.policy_max_grad_norm)
            ))
            critic_norms.append(float(
                nn.utils.clip_grad_norm_(model.critic_parameters(), config.critic_max_grad_norm)
            ))
            optimizer.step()

            with torch.no_grad():
                totals["policy_loss"] += float(policy_loss)
                # Unweighted, so the reading means the same with the weighting on or off.
                totals["value_loss"] += float(squared_errors.mean())
                totals["entropy"] += float(entropy)
                # Schulman's low-variance KL estimator; stays non-negative.
                totals["approx_kl"] += float(((ratio - 1) - log_ratio).mean())
                totals["clip_fraction"] += float(
                    ((ratio - 1).abs() > config.clip_epsilon).float().mean()
                )
            steps += 1

    stats = {name: value / max(steps, 1) for name, value in totals.items()}
    for name, norms, threshold in (
        ("policy", policy_norms, config.policy_max_grad_norm),
        ("critic", critic_norms, config.critic_max_grad_norm),
    ):
        stats[f"grad_norm_{name}"], stats[f"grad_norm_{name}_sd"] = _mean_and_sd(norms)
        stats[f"grad_clipped_{name}"] = sum(n > threshold for n in norms) / max(steps, 1)
    stats["grad_steps"] = float(steps)
    stats["adv_mean"] = raw_mean
    stats["adv_std"] = raw_std
    return stats


def _mean_and_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    return mean, math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


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
            "obs_dim": model.obs_dim,
            "action_dim": ACTION_DIM,
            "feature_version": model.feature_version,
            **model.shape,
            # The per-player encoder of the equity network the model was built with; its
            # weights are in `model`'s state, frozen.
            "equity": model.equity,
            "iteration": iteration,
            # On the CPU whatever the device it was trained on, so a model published by a GPU
            # worker loads anywhere.
            "model": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
            "optimizer": None if optimizer is None else optimizer.state_dict(),
            "metadata": metadata or {},
        },
        path,
    )


class IncompatibleCheckpointError(ValueError):
    """The checkpoint was trained against a different observation encoding."""


def check_compatible(checkpoint: dict[str, Any], source: str = "checkpoint") -> None:
    """Whether this build can serve the checkpoint: any feature version it still encodes
    (`SUPPORTED_VERSIONS`), at the dimension that version has."""
    saved_version = checkpoint.get("feature_version", 0)
    if saved_version not in SUPPORTED_VERSIONS:
        raise IncompatibleCheckpointError(
            f"{source} was trained on feature version {saved_version}, this build serves "
            f"{SUPPORTED_VERSIONS} -- a different encoding, so the weights would read the "
            f"wrong thing from every slot"
        )
    expected = observation_dim(saved_version)
    if checkpoint.get("obs_dim") != expected or checkpoint.get("action_dim") != ACTION_DIM:
        raise IncompatibleCheckpointError(
            f"{source} was trained on obs_dim={checkpoint.get('obs_dim')} "
            f"action_dim={checkpoint.get('action_dim')}, but feature version {saved_version} "
            f"is {expected}/{ACTION_DIM} -- the encoding changed since it was saved"
        )
    if any(key not in checkpoint for key in (*SHAPE_KEYS, "equity")) or checkpoint["equity"] is None:
        raise IncompatibleCheckpointError(
            f"{source} was saved before every network read the cards through an equity "
            "encoder, and its weights are named for the old layout, so it cannot be "
            "loaded into this build's network"
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
    if checkpoint["feature_version"] != model.feature_version:
        raise IncompatibleCheckpointError(
            f"{path} reads feature version {checkpoint['feature_version']}, this network "
            f"{model.feature_version}: their inputs are not the same, so the weights cannot "
            "be exchanged"
        )
    if checkpoint["equity"] != model.equity:
        raise IncompatibleCheckpointError(
            f"{path} was built with equity encoder {checkpoint['equity']}, this network "
            f"with {model.equity}: models built on different encoders cannot exchange weights"
        )
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and checkpoint["optimizer"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def checkpoint_shape(checkpoint: dict[str, Any]) -> dict[str, int]:
    """The network shape a checkpoint was saved with (after `check_compatible`)."""
    return {key: int(checkpoint[key]) for key in SHAPE_KEYS}


def build_model_from_checkpoint(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> tuple[PokerActorCritic, dict[str, Any]]:
    """Rebuild a saved network at the shape it was trained with, frozen for
    inference. Used to seat previously trained agents as opponents."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    check_compatible(checkpoint, str(path))
    model = PokerActorCritic(
        **checkpoint_shape(checkpoint),
        equity=checkpoint["equity"],
        feature_version=checkpoint["feature_version"],
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint

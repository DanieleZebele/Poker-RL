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
from pokerlab.rl.policy import CRITIC_OUTPUT_INIT, SHAPE_KEYS, PokerActorCritic, normalise_critic
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
    # The policy's gradient-norm clip. The critic's gradient is not clipped: the two networks
    # share no weight, so this clip never touches it, and its norm is only measured.
    policy_max_grad_norm: float = 0.5
    normalize_advantages: bool = True
    # Dual-clip PPO (Ye et al. 2020; AlphaHoldem's "trinal clip"): with a negative advantage the
    # surrogate is bounded below at `dual_clip * advantage`, so a ratio that has grown large
    # cannot make one minibatch push without limit. 0 = off (the plain clip); it must be > 1.
    dual_clip: float = 0.0
    # Soft shrink and perturb (Ash & Adams 2020; Juliani & Ash 2024 for PPO): after every step
    # each weight of a `Linear` becomes `(1 - b) * w + b * fresh`, `fresh` drawn as PyTorch
    # initialises the layer, against the loss of plasticity of a network trained on and on.
    # 0 = off.
    shrink_perturb: float = 0.0

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
        if self.dual_clip and self.dual_clip <= 1:
            raise ValueError(f"the dual clip bounds the ratio from above and must be > 1, not {self.dual_clip}")
        if not 0 <= self.shrink_perturb < 1:
            raise ValueError(f"shrink_perturb is a share in [0, 1), not {self.shrink_perturb}")
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

    return TrainingBatch(
        features=torch.tensor(features, dtype=torch.float32, device=device),
        masks=torch.tensor(masks, dtype=torch.bool, device=device),
        actions=torch.tensor(actions, dtype=torch.int64, device=device),
        old_log_probs=torch.tensor(old_log_probs, dtype=torch.float32, device=device),
        advantages=torch.tensor(advantages, dtype=torch.float32, device=device),
        returns=torch.tensor(returns, dtype=torch.float32, device=device),
        critic_extra=torch.tensor(critic_extra, dtype=torch.float32, device=device),
    )


def grad_norm(parameters: list[nn.Parameter]) -> float:
    """The norm of the gradient over `parameters`, measured and left as it is."""
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads])))


def ppo_update(
    model: PokerActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: TrainingBatch,
    config: PPOConfig,
) -> dict[str, float]:
    """One PPO update over a batch. Returns diagnostics, averaged per minibatch.

    The policy's gradient is clipped to `policy_max_grad_norm` and reported as the mean norm
    before the cut, its sd over the steps and the share of steps cut (`grad_norm_policy`,
    `grad_norm_policy_sd`, `grad_clipped_policy`, see `rl/grad_log.py`): the mean alone does
    not say how often the cut fires. The critic's is not clipped; its norm is reported the
    same way, without the share (`grad_norm_critic`, `grad_norm_critic_sd`)."""
    advantages = batch.advantages
    # Raw statistics, read before normalising: after it they are 0 and 1 by
    # construction, which says nothing about the scale the GAE produced.
    raw_mean = float(advantages.mean())
    raw_std = float(advantages.std()) if len(batch) > 1 else 0.0
    if config.normalize_advantages and len(batch) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0,
              "clip_fraction": 0.0}
    steps = 0
    # The norm of every step, per network: the policy's before its cut (what
    # `clip_grad_norm_` returns), the critic's as it is.
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
            logits = model.policy_logits(batch.features[index], batch.masks[index])
            returns = batch.returns[index]
            values, critic_losses = model.critic_terms(
                batch.features[index], batch.critic_extra[index], returns
            )
            distribution = torch.distributions.Categorical(logits=logits)

            log_probs = distribution.log_prob(batch.actions[index])
            log_ratio = log_probs - batch.old_log_probs[index]
            ratio = log_ratio.exp()
            minibatch_advantages = advantages[index]

            policy_loss = -clipped_objective(
                ratio, minibatch_advantages, config.clip_epsilon, config.dual_clip
            ).mean()
            value_loss = critic_losses.mean()
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
            critic_norms.append(grad_norm(model.critic_parameters()))
            optimizer.step()
            if config.shrink_perturb:
                shrink_and_perturb(model, config.shrink_perturb)

            with torch.no_grad():
                totals["policy_loss"] += float(policy_loss)
                # The squared error of the value, so the reading means the same with either
                # kind of critic.
                totals["value_loss"] += float(((values - returns) ** 2).mean())
                totals["entropy"] += float(entropy)
                # Schulman's low-variance KL estimator; stays non-negative.
                totals["approx_kl"] += float(((ratio - 1) - log_ratio).mean())
                totals["clip_fraction"] += float(
                    ((ratio - 1).abs() > config.clip_epsilon).float().mean()
                )
            steps += 1

    stats = {name: value / max(steps, 1) for name, value in totals.items()}
    stats["grad_norm_policy"], stats["grad_norm_policy_sd"] = _mean_and_sd(policy_norms)
    stats["grad_clipped_policy"] = sum(n > config.policy_max_grad_norm for n in policy_norms) / max(steps, 1)
    stats["grad_norm_critic"], stats["grad_norm_critic_sd"] = _mean_and_sd(critic_norms)
    stats["grad_steps"] = float(steps)
    stats["adv_mean"] = raw_mean
    stats["adv_std"] = raw_std
    return stats


def clipped_objective(ratio: Tensor, advantages: Tensor, epsilon: float, dual_clip: float = 0.0) -> Tensor:
    """PPO's clipped surrogate per decision (to be maximised); with `dual_clip` (> 1) a negative
    advantage's term is bounded below at `dual_clip * advantage`."""
    surrogate = torch.min(ratio * advantages, ratio.clamp(1 - epsilon, 1 + epsilon) * advantages)
    if dual_clip:
        surrogate = torch.where(advantages < 0, torch.max(surrogate, dual_clip * advantages), surrogate)
    return surrogate


@torch.no_grad()
def shrink_and_perturb(model: PokerActorCritic, share: float) -> None:
    """Every trainable `Linear` of the policy and the critic moved `share` of the way towards a
    fresh initialisation of itself (the frozen encoders are left alone)."""
    for module in (model.trunk, model.policy_head, model.value_trunk, model.value_head):
        for layer in module.modules():
            if isinstance(layer, nn.Linear):
                bound = 1 / math.sqrt(layer.in_features)  # PyTorch's own `Linear` initialisation
                for parameter in (layer.weight, layer.bias):
                    fresh = torch.empty_like(parameter).uniform_(-bound, bound)
                    parameter.mul_(1 - share).add_(fresh, alpha=share)


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
            # None for the scalar critic, else its support (`rl/policy.py`, `CRITIC_KEYS`).
            "critic": model.critic,
            # How a classifier's output layer was started (`policy.CRITIC_OUTPUT_INIT`).
            "critic_output_init": CRITIC_OUTPUT_INIT if model.critic is not None else None,
            # Feature version 5: the belief network's shape and the fingerprint of the file it
            # came from (`rl/belief.py`); its weights are in `model`'s state, frozen.
            "belief": model.belief,
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
    if (saved_version == 5) != (checkpoint.get("belief") is not None):
        raise IncompatibleCheckpointError(
            f"{source} reads feature version {saved_version} and "
            f"{'has no' if saved_version == 5 else 'has a'} belief network: only version 5 has one"
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
    if checkpoint.get("belief") != model.belief:
        raise IncompatibleCheckpointError(
            f"{path} was built with belief network {checkpoint.get('belief')}, this network with "
            f"{model.belief}: a policy trained on one belief's vectors cannot read another's"
        )
    state = checkpoint["model"]
    stuck = model.critic is not None and checkpoint.get("critic_output_init") != CRITIC_OUTPUT_INIT
    if critic_of(checkpoint) != model.critic or stuck:
        # Another kind of critic (a scalar one, or bins over another support): everything
        # carries over but the critic's output layer, which means something else and starts
        # afresh. The policy is the same network whatever the critic, so a child keeps its
        # parent's play. The optimizer's state is shaped for the old layer and is dropped.
        # The same for a classifier saved before `CRITIC_OUTPUT_INIT`: its output layer began
        # where it barely learns and has stayed there.
        output = critic_output_keys(model)
        state = {key: value for key, value in state.items() if key not in output}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if set(missing) != output or unexpected:
            raise IncompatibleCheckpointError(
                f"{path}: loading it into a network with another critic left {missing} "
                f"missing and {unexpected} unexpected"
            )
        checkpoint["critic_head_reset"] = True
        return checkpoint
    model.load_state_dict(state)
    if optimizer is not None and checkpoint["optimizer"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def critic_of(checkpoint: dict[str, Any]) -> dict[str, Any] | None:
    """The critic a checkpoint was saved with: None (the scalar one) for every checkpoint
    saved before the classifier existed."""
    return normalise_critic(checkpoint.get("critic"))


def critic_output_keys(model: PokerActorCritic) -> set[str]:
    """The state keys of the critic's output layer: the one part of the network that depends
    on the kind of critic."""
    last = len(model.value_head) - 1
    return {f"value_head.{last}.weight", f"value_head.{last}.bias"}


def checkpoint_shape(checkpoint: dict[str, Any]) -> dict[str, int]:
    """The network shape a checkpoint was saved with (after `check_compatible`); `residual` is
    0 for every checkpoint saved before it existed."""
    return {**{key: int(checkpoint[key]) for key in SHAPE_KEYS}, "residual": int(checkpoint.get("residual", 0))}


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
        critic=critic_of(checkpoint),
        belief=checkpoint.get("belief"),
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint

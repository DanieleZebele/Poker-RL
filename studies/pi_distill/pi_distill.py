"""Privileged-information distillation in PPO (Penaloza et al., "Privileged Information
Distillation for Language Models", 2026) -- a study, nothing in `src/` uses it.

The privileged information is every player's equity against the hands really dealt: the nine
numbers the critic already reads (`critic_extra`, slot 0 the player deciding, 0 for a folded
seat). The **teacher** is the policy itself with that one more input, added to the first layer of
the trunk through a projection that starts at zero: every other weight is shared (the paper's
"same model", one set of parameters), the teacher starts as the student, and the **student** is
exactly the network that plays and is published -- a model trained here is an ordinary
checkpoint.

The study is `poker-train` itself with two things replaced: the PPO update (`distill_update`, the
production update plus the terms below) and the publication (nothing reaches the shared store; the
model stays in `--scratch-dir`). Every other flag is `poker-train`'s.

Who plays the learner's seats (`--pi-actor`):
  - `student` (OPSD, the paper's on-policy variant): hands as in production. The loss adds the
    teacher's PPO term off-policy on the student's actions (ratio pi_T(a|s,I) / pi_old(a|s), weight
    `--pi-teacher-weight`), which is how the teacher learns to read the equities, and
    `--pi-beta-student` x KL(pi_S || sg pi_T), the student pulled to the teacher.
  - `teacher` (pi-Distill): the teacher plays, reading the equities at every decision; its PPO term
    is on-policy, the student's is off-policy on the teacher's actions (ratio pi_S / pi_T_old,
    clipped) with weight `--pi-student-weight`, and the student's KL is the forward one,
    KL(sg pi_T || pi_S) (the paper's eq. 3).
In both `--pi-beta-teacher` x KL(pi_T || sg pi_S) keeps the teacher near the student. The defaults
(student plays, student weight 1, everything else 0) are production PPO.

`--pi-separate-teacher`: the teacher is a network of its own instead, a copy of the student's trunk
and head taken at the start (plus the projection), so it still starts as the student; its terms then
reach the student only through the KL, and its gradient norm is clipped apart from the student's.

    OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH=src:studies/pi_distill \\
        python studies/pi_distill/pi_distill.py --pi-actor student --pi-teacher-weight 1 \\
        --pi-beta-student 0.1 --pi-beta-teacher 0.1 -- <poker-train flags>

**The critic in hindsight** (`--pi-critic-board`, the other use of privileged information: it
never reaches the policy, it only takes luck out of the signal the policy learns from). The critic
reads every player's equity; `now` (production) computes it on the board as it stands, `real` on
the board this hand will have once all five cards are out (`poker-train --critic-runout`: the deck
is already shuffled, so that is who wins a showdown) and `random` on the visible cards completed at
random, the control; `both` gives the critic the two sets side by side (18 numbers in place of 9:
the first layer of its trunk is widened when the critic is made, and a student is saved with a
fresh layer of the usual width, since only its policy is ever played). A baseline may read anything
the action does not change: the cards still in the deck and the other players' cards qualify, so
the policy's gradient keeps its expectation and only its variance moves.

Every iteration prints a `distill:` line: KL(S||T) overall and by street, KL(T||S), and the
teacher's and the student's edge -- the importance-weighted mean of the normalised advantage under
each policy, over the decisions of the batch, before the update (0 for the policy that played:
what the other would have gained, one step, by playing its own probabilities).
"""

from __future__ import annotations

import argparse
import copy
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor, nn

from pokerlab.players import rl_agent
from pokerlab.players.rl_agent import DealView, PolicyDecision, RLAgentPlayer, deal_view
from pokerlab.rl.features import CARDS_DIM, STREET_DIM, card_index
from pokerlab.rl.policy import (
    BOARD_PLANE,
    EQUITY_SLOTS,
    HOLE_PLANE,
    MASK_FILL,
    PLANE,
    PokerActorCritic,
)
from pokerlab.rl.ppo import (
    PPOConfig,
    TrainingBatch,
    _mean_and_sd,
    clipped_objective,
    grad_norm,
    save_checkpoint,
    shrink_and_perturb,
)

STREETS = ("preflop", "flop", "turn", "river")
CRITIC_BOARDS = ("now", "real", "random", "both")
# The importance weight of the edge estimate is cut here: a policy that gives an action fifty
# times the probability the player gave it would otherwise decide the mean by itself.
EDGE_RATIO_CAP = 10.0


@dataclass
class Settings:
    actor: str = "student"
    student_weight: float = 1.0
    teacher_weight: float = 0.0
    beta_student: float = 0.0
    beta_teacher: float = 0.0
    teacher_clip: float | None = None   # None: the run's clip_epsilon
    # The projection's learning rate over the run's (which the trainer sets on every group each
    # iteration): it starts at zero, and at the run's 1e-4 a few hundred steps leave it there.
    teacher_lr_scale: float = 1.0
    # A control: the teacher reads the equities of another decision of the batch (a permutation
    # drawn per update), the same distribution and no information about this one.
    placebo: bool = False
    # The teacher is a copy of the student's trunk and head, not the student's own weights.
    separate: bool = False
    # Which board the critic's equities are computed on: `now` (as it stands), `real` (the
    # five cards this hand will have), `random` (the visible ones completed at random), `both`
    # (now and real, side by side).
    critic_board: str = "now"
    # Iterations after whose update the student is written to `save_dir` (student-iNNNN.pt).
    save_at: tuple[int, ...] = ()
    save_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.actor not in ("student", "teacher"):
            raise ValueError(f"actor must be student or teacher, got {self.actor}")
        if min(self.student_weight, self.teacher_weight, self.beta_student, self.beta_teacher) < 0:
            raise ValueError("weights and betas must not be negative")
        if self.placebo and self.actor == "teacher":
            raise ValueError("the placebo is a control of the student-played runs")
        if self.save_at and self.save_dir is None:
            raise ValueError("save_at needs a save_dir")
        if self.critic_board not in CRITIC_BOARDS:
            raise ValueError(f"critic_board must be one of {CRITIC_BOARDS}, got {self.critic_board}")
        if self.critic_board != "now" and self.uses_teacher:
            raise ValueError("the critic in hindsight is a study of its own: no teacher with it")

    @property
    def uses_teacher(self) -> bool:
        return self.actor == "teacher" or bool(
            self.teacher_weight or self.beta_student or self.beta_teacher
        )


class Teacher(nn.Module):
    """What the teacher has of its own: the projection of the equities into the first layer of the
    trunk, zero at the start and with no bias (an all-zero input adds nothing), and, when
    `separate`, a copy of the student's trunk and policy head. Either way it starts as the student."""

    def __init__(self, model: PokerActorCritic, separate: bool = False) -> None:
        super().__init__()
        first = model.trunk[0]
        self.projection = nn.Linear(EQUITY_SLOTS, first.out_features, bias=False)
        nn.init.zeros_(self.projection.weight)
        self.separate = separate
        if separate:
            self.trunk = copy.deepcopy(model.trunk)
            self.head = copy.deepcopy(model.policy_head)
        self.to(first.weight.device)

    def copy_student(self, model: PokerActorCritic) -> None:
        """Take the student's trunk and head as they are now (in place: the optimizer keeps its
        parameters). `poker-train` makes the critic, hence the teacher, before it loads the
        checkpoint it resumes from, so the copy is taken again at the teacher's first use."""
        if self.separate:
            self.trunk.load_state_dict(model.trunk.state_dict())
            self.head.load_state_dict(model.policy_head.state_dict())

    def copied_parameters(self) -> list[nn.Parameter]:
        if not self.separate:
            return []
        return list(self.trunk.parameters()) + list(self.head.parameters())


def make_teacher(model: PokerActorCritic, separate: bool = False) -> Teacher:
    return Teacher(model, separate)


def both_logits(
    model: PokerActorCritic, teacher: Teacher, features: Tensor, mask: Tensor, privileged: Tensor
) -> tuple[Tensor, Tensor]:
    """The student's and the teacher's masked logits `(B, ACTION_DIM)`, the shared part computed
    once (the frozen card encoder, and the trunk too unless the teacher is separate). The student's
    are `model.policy_logits`."""
    latent = model.equity_encoder(
        features[:, HOLE_PLANE * PLANE : (HOLE_PLANE + 1) * PLANE],
        features[:, BOARD_PLANE * PLANE : (BOARD_PLANE + 1) * PLANE],
    )
    inputs = torch.cat([features[:, CARDS_DIM:], latent], dim=-1)
    first = model.trunk[0](inputs)
    student = model.policy_head(model.trunk[1:](first))
    added = teacher.projection(privileged)
    if teacher.separate:
        hidden = teacher.trunk[1:](teacher.trunk[0](inputs) + added)
        logits = teacher.head(hidden)
    else:
        logits = model.policy_head(model.trunk[1:](first + added))
    return student.masked_fill(~mask, MASK_FILL), logits.masked_fill(~mask, MASK_FILL)


def kl(log_p: Tensor, log_q: Tensor) -> Tensor:
    """KL(p || q) per row, from log-probabilities; a masked bin (probability 0) adds nothing."""
    return (log_p.exp() * (log_p - log_q)).sum(dim=-1)


def clipped_surrogate(ratio: Tensor, advantages: Tensor, epsilon: float) -> Tensor:
    return -torch.min(ratio * advantages, ratio.clamp(1 - epsilon, 1 + epsilon) * advantages).mean()


def street_of(features: Tensor) -> Tensor:
    return features[:, CARDS_DIM : CARDS_DIM + STREET_DIM].argmax(dim=-1)


@dataclass
class Distiller:
    """The state the replaced functions share: the settings, the teacher (made when the model
    is), and the critic's equity function (what the teacher reads when it plays)."""

    settings: Settings
    teacher: Teacher | None = None
    model: PokerActorCritic | None = None
    equity: object | None = None
    optimizer_ids: set[int] = field(default_factory=set)
    iteration: int = 0
    teacher_playing: bool = False
    copied: bool = False
    placebo_rng: torch.Generator = field(default_factory=lambda: torch.Generator().manual_seed(12345))

    def attach(self, model: PokerActorCritic) -> None:
        if self.teacher is None:
            self.model = model
            self.teacher = make_teacher(model, self.settings.separate)

    def ready(self) -> Teacher:
        """The teacher, its copy of the student taken the first time it is used."""
        if not self.copied:
            self.teacher.copy_student(self.model)
            self.copied = True
        return self.teacher

    def ensure_in(self, optimizer: torch.optim.Optimizer) -> None:
        """The teacher trains with the run's optimizer, at the run's learning rate (the trainer
        sets it on every group before the update): its projection at `teacher_lr_scale` times
        that, the copied trunk and head of a separate teacher at that rate."""
        projection = list(self.teacher.projection.parameters())
        copied = self.teacher.copied_parameters()
        if id(optimizer) not in self.optimizer_ids:
            optimizer.add_param_group({"params": projection})
            if copied:
                optimizer.add_param_group({"params": copied})
            self.optimizer_ids.add(id(optimizer))
        projection_ids = set(map(id, projection))
        teacher_ids = set(map(id, self.teacher.parameters()))
        base = next(g["lr"] for g in optimizer.param_groups if not teacher_ids & set(map(id, g["params"])))
        for group in optimizer.param_groups:
            ids = set(map(id, group["params"]))
            if projection_ids & ids:
                group["lr"] = base * self.settings.teacher_lr_scale
            elif teacher_ids & ids:
                group["lr"] = base

    def privileged(self, views) -> Tensor:
        return torch.tensor(self.equity(views), dtype=torch.float32)

    def policy_fn(self):
        """The teacher as the learner's policy: it reads the equities of the decision's deal,
        which the patched `RLAgentPlayer.act` puts on it just before asking."""
        model = self.model
        self.teacher_playing = True

        def teacher_policy_fn(features, mask, bias=None, temperature=1.0) -> PolicyDecision:
            if bias is not None:
                raise ValueError("the teacher is only ever the learner, which has no style")
            view = teacher_policy_fn.view
            if view is None:
                raise ValueError("the teacher was asked without the deal (no deal_holes)")
            teacher_policy_fn.view = None
            with torch.no_grad():
                _, logits = both_logits(
                    model, self.ready(),
                    torch.tensor([features], dtype=torch.float32),
                    torch.tensor([mask], dtype=torch.bool),
                    self.privileged([view]),
                )
                distribution = torch.distributions.Categorical(logits=logits)
                action = distribution.sample()
            return PolicyDecision(int(action.item()), float(distribution.log_prob(action).item()), 0.0)

        teacher_policy_fn.view = None
        teacher_policy_fn.privileged = True
        teacher_policy_fn.feature_version = model.feature_version
        return teacher_policy_fn


def _patched_act(original):
    def act(self, observation, legal_actions):
        if getattr(self._policy_fn, "privileged", False):
            if self._deal_holes is None:
                raise ValueError("a privileged policy needs the table's hole cards")
            self._policy_fn.view = deal_view(observation, self._deal_holes())
        return original(self, observation, legal_actions)

    act.patched = True
    return act


def distill_update(
    distiller: Distiller,
    model: PokerActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: TrainingBatch,
    config: PPOConfig,
) -> dict[str, float]:
    """`ppo.ppo_update` with the teacher's terms. With the default settings it is the production
    update, step for step (pinned by a test)."""
    s = distiller.settings
    if s.actor == "teacher" and not distiller.teacher_playing:
        raise ValueError("--pi-actor teacher, but the student played: one device only (--update-device same)")
    teacher_on = s.uses_teacher
    teacher_net = distiller.ready() if teacher_on else None
    if teacher_on:
        distiller.ensure_in(optimizer)
    privileged = batch.critic_extra
    if teacher_on and s.placebo:
        privileged = privileged[torch.randperm(len(batch), generator=distiller.placebo_rng)]
    advantages = batch.advantages
    raw_mean = float(advantages.mean())
    raw_std = float(advantages.std()) if len(batch) > 1 else 0.0
    if config.normalize_advantages and len(batch) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    before = diagnostics(model, teacher_net, batch, advantages, privileged) if teacher_on else {}

    totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0,
              "clip_fraction": 0.0, "teacher_loss": 0.0, "kl_st": 0.0, "kl_ts": 0.0}
    steps = 0
    policy_norms: list[float] = []
    critic_norms: list[float] = []
    full = len(batch) - len(batch) % config.minibatch_size
    stop = full if full else len(batch)
    # A shared teacher's projection is clipped with the weights it feeds; a separate teacher is
    # a network of its own and is clipped apart, so its gradients never shorten the student's step.
    separate = teacher_on and teacher_net.separate
    policy_params = model.policy_parameters()
    teacher_params = list(teacher_net.parameters()) if teacher_on else []
    if not separate:
        policy_params, teacher_params = policy_params + teacher_params, []
    teacher_clip = s.teacher_clip if s.teacher_clip is not None else config.clip_epsilon

    for _ in range(config.epochs):
        permutation = torch.randperm(len(batch), device=batch.features.device)
        for start in range(0, stop, config.minibatch_size):
            index = permutation[start : start + config.minibatch_size]
            features, masks = batch.features[index], batch.masks[index]
            if teacher_on:
                student_logits, teacher_logits = both_logits(
                    model, teacher_net, features, masks, privileged[index]
                )
            else:
                student_logits, teacher_logits = model.policy_logits(features, masks), None
            returns = batch.returns[index]
            values, critic_losses = model.critic_terms(features, batch.critic_extra[index], returns)
            student = torch.distributions.Categorical(logits=student_logits)
            minibatch_advantages = advantages[index]
            old = batch.old_log_probs[index]
            actions = batch.actions[index]

            log_ratio = student.log_prob(actions) - old
            ratio = log_ratio.exp()
            policy_loss = -clipped_objective(ratio, minibatch_advantages, config.clip_epsilon, config.dual_clip).mean()
            value_loss = critic_losses.mean()
            entropy = student.entropy().mean()
            loss = (
                s.student_weight * policy_loss
                + config.value_coefficient * value_loss
                - config.entropy_coefficient * entropy
            )
            if teacher_on:
                teacher = torch.distributions.Categorical(logits=teacher_logits)
                teacher_ratio = (teacher.log_prob(actions) - old).exp()
                teacher_loss = clipped_surrogate(teacher_ratio, minibatch_advantages, teacher_clip)
                log_s = torch.log_softmax(student_logits, dim=-1)
                log_t = torch.log_softmax(teacher_logits, dim=-1)
                if s.actor == "student":   # OPSD: reverse KL, the student's own mass
                    student_kl = kl(log_s, log_t.detach()).mean()
                else:                      # pi-Distill: forward KL, the teacher's mass
                    student_kl = kl(log_t.detach(), log_s).mean()
                teacher_kl = kl(log_t, log_s.detach()).mean()
                loss = (
                    loss
                    + s.teacher_weight * (teacher_loss - config.entropy_coefficient * teacher.entropy().mean())
                    + s.beta_student * student_kl
                    + s.beta_teacher * teacher_kl
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            policy_norms.append(float(nn.utils.clip_grad_norm_(policy_params, config.policy_max_grad_norm)))
            if teacher_params:
                nn.utils.clip_grad_norm_(teacher_params, config.policy_max_grad_norm)
            critic_norms.append(grad_norm(model.critic_parameters()))
            optimizer.step()
            if config.shrink_perturb:
                shrink_and_perturb(model, config.shrink_perturb)

            with torch.no_grad():
                totals["policy_loss"] += float(policy_loss)
                totals["value_loss"] += float(((values - returns) ** 2).mean())
                totals["entropy"] += float(entropy)
                totals["approx_kl"] += float(((ratio - 1) - log_ratio).mean())
                totals["clip_fraction"] += float(((ratio - 1).abs() > config.clip_epsilon).float().mean())
                if teacher_on:
                    totals["teacher_loss"] += float(teacher_loss)
                    totals["kl_st"] += float(kl(log_s, log_t).mean())
                    totals["kl_ts"] += float(kl(log_t, log_s).mean())
            steps += 1

    stats = {name: value / max(steps, 1) for name, value in totals.items()}
    stats["grad_norm_policy"], stats["grad_norm_policy_sd"] = _mean_and_sd(policy_norms)
    stats["grad_clipped_policy"] = sum(n > config.policy_max_grad_norm for n in policy_norms) / max(steps, 1)
    stats["grad_norm_critic"], stats["grad_norm_critic_sd"] = _mean_and_sd(critic_norms)
    stats["grad_steps"] = float(steps)
    stats["adv_mean"] = raw_mean
    stats["adv_std"] = raw_std
    distiller.iteration += 1
    if distiller.iteration in s.save_at:
        s.save_dir.mkdir(parents=True, exist_ok=True)
        saved = playable_copy(model) if s.critic_board == "both" else model
        save_checkpoint(s.save_dir / f"student-i{distiller.iteration:04}.pt", saved, iteration=distiller.iteration)
    if teacher_on:
        print(format_line(distiller.iteration, before, stats), flush=True)
    stats.update({f"before_{k}": v for k, v in before.items() if not isinstance(v, dict)})
    return stats


def diagnostics(model, teacher, batch: TrainingBatch, advantages: Tensor, privileged: Tensor) -> dict:
    """Before the update, over the whole batch: how far apart the two policies are (by street),
    and the one-step edge each would have over the policy that played."""
    kl_st, kl_ts, edge_t, edge_s, streets = [], [], [], [], []
    with torch.no_grad():
        for start in range(0, len(batch), 8192):
            sl = slice(start, start + 8192)
            student_logits, teacher_logits = both_logits(
                model, teacher, batch.features[sl], batch.masks[sl], privileged[sl]
            )
            log_s = torch.log_softmax(student_logits, -1)
            log_t = torch.log_softmax(teacher_logits, -1)
            kl_st.append(kl(log_s, log_t))
            kl_ts.append(kl(log_t, log_s))
            actions = batch.actions[sl, None]
            old = batch.old_log_probs[sl]
            for log, out in ((log_t, edge_t), (log_s, edge_s)):
                weight = (log.gather(1, actions).squeeze(1) - old).exp().clamp(max=EDGE_RATIO_CAP)
                out.append(weight * advantages[sl])
            streets.append(street_of(batch.features[sl]))
    kl_st, kl_ts = torch.cat(kl_st), torch.cat(kl_ts)
    street = torch.cat(streets)
    return {
        "kl_st": float(kl_st.mean()),
        "kl_ts": float(kl_ts.mean()),
        "edge_teacher": float(torch.cat(edge_t).mean()),
        "edge_student": float(torch.cat(edge_s).mean()),
        "kl_st_by_street": {
            name: float(kl_st[street == i].mean()) for i, name in enumerate(STREETS) if (street == i).any()
        },
    }


def format_line(iteration: int, before: dict, stats: dict) -> str:
    by_street = " ".join(f"{name} {value:.3f}" for name, value in before["kl_st_by_street"].items())
    return (
        f"distill {iteration:4}: kl(S||T) {before['kl_st']:.4f} [{by_street}]  kl(T||S) {before['kl_ts']:.4f}"
        f"  edge teacher {before['edge_teacher']:+.4f} student {before['edge_student']:+.4f}"
        f"  | teacher loss {stats['teacher_loss']:+.4f}"
    )


def parse_line(line: str) -> dict | None:
    """The numbers of a `distill` line, for whoever reads the logs back."""
    head = re.match(r"distill +(\d+): ", line)
    if head is None:
        return None
    out: dict[str, float] = {"iteration": int(head.group(1))}
    for key, pattern in (
        ("kl_st", r"kl\(S\|\|T\) ([-+.\d]+)"),
        ("kl_ts", r"kl\(T\|\|S\) ([-+.\d]+)"),
        ("edge_teacher", r"edge teacher ([-+.\d]+)"),
        ("edge_student", r"edge teacher [-+.\d]+ student ([-+.\d]+)"),
    ):
        found = re.search(pattern, line)
        if found:
            out[key] = float(found.group(1))
    streets = re.search(r"\[([^\]]*)\]", line)
    if streets:
        words = streets.group(1).split()
        out.update({f"kl_st_{name}": float(value) for name, value in zip(words[::2], words[1::2], strict=True)})
    return out


def hindsight_board(visible: tuple[int, ...], runout: tuple[int, ...], holes: set[int]) -> tuple[int, ...]:
    """The control's board: the visible cards (the first of the hand's `runout`), completed at
    random from the cards in no hand. Seeded by the deal, so it is the same at every decision of
    a hand and a run reproduces."""
    if visible != runout[: len(visible)]:
        raise ValueError(f"the visible board {visible} is not the start of the runout {runout}")
    rest = [card for card in range(52) if card not in holes and card not in runout[: len(visible)]]
    random.Random(sum(c << (6 * i) for i, c in enumerate(runout))).shuffle(rest)
    return visible + tuple(rest[: 5 - len(visible)])


def _random_view(original):
    """`rl_agent.deal_view` for the control: the runout the collector hands over is replaced by
    a random completion of the visible board."""

    def view(observation, hole_cards, runout=None) -> DealView:
        if runout is None:
            raise ValueError("the control needs the runout: run with --critic-runout")
        plain = original(observation, hole_cards)
        holes = {card_index(card) for pair in hole_cards.values() for card in pair}
        board = tuple(card_index(card) for card in runout)
        return DealView(plain.holes, hindsight_board(plain.board, board, holes))

    return view


@dataclass(frozen=True)
class BothView(DealView):
    """A deal view whose board is the runout, with how many of its cards are out."""

    visible: int = 0


def _both_view(original):
    def view(observation, hole_cards, runout=None) -> DealView:
        if runout is None:
            raise ValueError("`both` needs the runout: run with --critic-runout")
        full = original(observation, hole_cards, runout)
        return BothView(full.holes, full.board, len(observation.community_cards))

    return view


def both_equity(equity):
    """The critic's 18 numbers: every seat's equity on the board as it stands, then on the
    hand's whole board."""

    def both(views):
        now = equity([DealView(v.holes, v.board[: v.visible]) for v in views])
        real = equity(list(views))
        return [a + b for a, b in zip(now, real, strict=True)]

    return both


def widen_critic(model: PokerActorCritic) -> None:
    """The critic's first layer reads 9 more numbers (fresh: the critic is trained from scratch)."""
    first = model.value_trunk[0]
    model.value_trunk[0] = nn.Linear(first.in_features + EQUITY_SLOTS, first.out_features).to(first.weight.device)


def playable_copy(model: PokerActorCritic) -> PokerActorCritic:
    """A copy whose critic's first layer has the usual width again (fresh): what a student is
    saved as, so it loads and plays like any model; its critic is never used."""
    clone = copy.deepcopy(model)
    first = clone.value_trunk[0]
    clone.value_trunk[0] = nn.Linear(first.in_features - EQUITY_SLOTS, first.out_features)
    return clone


# What `poker-train` does to the shared store and ledger, and the study never may: the model is
# not published (it stays in --scratch-dir), and no rating pass of the fleet is played.
SHARED_WRITERS = ("publish_model", "publish_snapshot", "record_sweep_observation", "run_elo_fill_in")
ARENA_WRITERS = ("run_population_sessions", "add_benchmark_candidates")


ITER_LINE = re.compile(r"iter +(\d+) +reward +([-+.\d]+) bb .*entropy ([.\d]+) +kl ([.\d]+)")


def summarise(text: str, last: int = 5) -> dict:
    """A run's log in a few numbers: iterations, the training reward (bb a hand, over the last
    `last` iterations: the teacher's when it played) and the entropy, and the `distill` lines'
    means over the same iterations."""
    iters = [m.groups() for m in map(ITER_LINE.match, text.splitlines()) if m]
    distills = [d for d in map(parse_line, text.splitlines()) if d]
    tail = iters[-last:]
    out = {"iterations": len(iters)}
    if tail:
        out["reward_bb"] = sum(float(t[1]) for t in tail) / len(tail)
        out["entropy"] = sum(float(t[2]) for t in tail) / len(tail)
    for key in ("kl_st", "kl_ts", "edge_teacher", "edge_student", "kl_st_preflop", "kl_st_river"):
        values = [d[key] for d in distills[-last:] if key in d]
        if values:
            out[key] = sum(values) / len(values)
    return out


def format_summary(rows: dict[str, dict]) -> str:
    keys = ("iterations", "reward_bb", "entropy", "kl_st", "kl_st_preflop", "kl_st_river", "kl_ts",
            "edge_teacher", "edge_student")
    lines = [f"{'run':24}" + "".join(f"{k:>14}" for k in keys)]
    for name, row in rows.items():
        cells = "".join(
            f"{row[k]:>14}" if k == "iterations" else (f"{row[k]:>14.4f}" if k in row else f"{'-':>14}")
            for k in keys
        )
        lines.append(f"{name:24}{cells}")
    return "\n".join(lines)


RUN_NAME = re.compile(r"(?P<arm>.+)-s(?P<seed>\d+)$")
STUDENT_NAME = re.compile(r"student-i(?P<iteration>\d+)\.pt$")


def group_ratings(results: list[dict]) -> dict[tuple[str, int], list[dict]]:
    """`rate_and_publish` results by (arm, iteration): a checkpoint is
    `<dir>/<arm>-s<seed>/students/student-i<iteration>.pt`."""
    groups: dict[tuple[str, int], list[dict]] = {}
    for result in results:
        path = Path(result["checkpoint"])
        run = RUN_NAME.match(path.parent.parent.name)
        student = STUDENT_NAME.match(path.name)
        if run is None or student is None:
            continue
        groups.setdefault((run["arm"], int(student["iteration"])), []).append(
            {**result, "seed": int(run["seed"])}
        )
    return groups


def mean_and_se(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, float("nan")
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, (variance / n) ** 0.5


def format_ratings(groups: dict[tuple[str, int], list[dict]], baseline: str = "base") -> str:
    """Per arm and iteration: bb/100 against the anchors and rating, mean +- se over the seeds, and
    the difference from the baseline arm at the same iteration (se of the difference of two
    independent means). Each seed's own number is listed too."""
    lines = [f"{'arm':12}{'iter':>5} {'seeds':>5} {'bb/100':>16} {'vs ' + baseline:>16} {'rating':>14}  per seme (bb/100)"]
    for (arm, iteration) in sorted(groups, key=lambda k: (k[1], k[0] != baseline, k[0])):
        rows = sorted(groups[(arm, iteration)], key=lambda r: r["seed"])
        bb, bb_se = mean_and_se([r["bb_per_100"] for r in rows])
        rating, rating_se = mean_and_se([r["rating"] for r in rows])
        diff = "-"
        base_rows = groups.get((baseline, iteration))
        if base_rows and arm != baseline:
            base, base_se = mean_and_se([r["bb_per_100"] for r in base_rows])
            diff = f"{bb - base:+.1f} +- {(bb_se ** 2 + base_se ** 2) ** 0.5:.1f}"
        seeds = " ".join(f"{r['bb_per_100']:+.1f}" for r in rows)
        lines.append(f"{arm:12}{iteration:>5} {len(rows):>5} {bb:>+9.1f} +- {bb_se:<4.1f} {diff:>16} "
                     f"{rating:>7.0f} +- {rating_se:<3.0f}  {seeds}")
    return "\n".join(lines)


def install(settings: Settings, train_module, arena_module=None) -> Distiller:
    """Replace, in `pokerlab.rl.train`, what the study changes: the update, the critic's
    factory (to make the teacher and keep the equity function), the learner's policy when the
    teacher plays, and everything that writes to the shared store or ledger (`SHARED_WRITERS`,
    and `ARENA_WRITERS` in `arena_module` when given)."""
    distiller = Distiller(settings)
    original_critic_fns = train_module.make_critic_fns
    original_policy_fn = train_module.make_policy_fn

    def make_critic_fns(model, equity_net, **kwargs):
        critic = original_critic_fns(model, equity_net, **kwargs)
        distiller.attach(model)
        distiller.equity = critic.equity
        if settings.critic_board == "both":
            widen_critic(model)
            critic = type(critic)(equity=both_equity(critic.equity), value=critic.value)
        return critic

    def make_policy_fn(model, **kwargs):
        # The pool's opponents are other networks. The first call on the learner's own network is
        # the collector's (the trainer's constructor); a later one -- a rated pass -- plays the
        # student, as a published model would. One device only: the network that plays must be
        # the one the teacher was made for.
        if (
            settings.actor == "teacher"
            and model is distiller.model
            and not getattr(make_policy_fn, "used", False)
        ):
            make_policy_fn.used = True
            return distiller.policy_fn()
        return original_policy_fn(model, **kwargs)

    def refused(name):
        def writer(*args, **kwargs):
            print(f"studio: {name} non eseguito (niente nello store né nel registro condiviso)", flush=True)
            if name == "run_elo_fill_in":
                return 0, 0
            return None

        return writer

    for name in SHARED_WRITERS:
        setattr(train_module, name, refused(name))
    if arena_module is not None:
        for name in ARENA_WRITERS:
            setattr(arena_module, name, refused(name))
    train_module.make_critic_fns = make_critic_fns
    train_module.make_policy_fn = make_policy_fn
    train_module.ppo_update = lambda model, optimizer, batch, config: distill_update(
        distiller, model, optimizer, batch, config
    )
    if not getattr(RLAgentPlayer.act, "patched", False):
        RLAgentPlayer.act = _patched_act(RLAgentPlayer.act)
    if settings.critic_board == "random":
        rl_agent.deal_view = _random_view(rl_agent.deal_view)
    if settings.critic_board == "both":
        rl_agent.deal_view = _both_view(rl_agent.deal_view)
    return distiller


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], add_help=True)
    parser.add_argument("--pi-actor", choices=("student", "teacher"), default="student")
    parser.add_argument("--pi-student-weight", type=float, default=1.0)
    parser.add_argument("--pi-teacher-weight", type=float, default=0.0)
    parser.add_argument("--pi-beta-student", type=float, default=0.0)
    parser.add_argument("--pi-beta-teacher", type=float, default=0.0)
    parser.add_argument("--pi-teacher-clip", type=float, default=None)
    parser.add_argument("--pi-teacher-lr-scale", type=float, default=1.0)
    parser.add_argument("--pi-placebo", action="store_true",
                        help="control: the teacher reads another decision's equities")
    parser.add_argument("--pi-separate-teacher", action="store_true",
                        help="the teacher is a copy of the student's trunk and head, not its weights")
    parser.add_argument("--pi-critic-board", choices=CRITIC_BOARDS, default="now",
                        help="the board the critic's equities are computed on: as it stands, the "
                        "real one once all five cards are out, or the visible one completed at random")
    parser.add_argument("--pi-save-at", type=int, nargs="*", default=[],
                        help="iterations after which the student is written to --pi-save-dir")
    parser.add_argument("--pi-save-dir", type=Path, default=None)
    parser.add_argument("--pi-save-teacher", type=Path, default=None,
                        help="where to write the teacher's own weights at the end")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "ratings":
        # `ratings FILE.json ...`: rate_and_publish results, by arm and iteration
        import json

        results = [r for f in argv[1:] for r in json.loads(Path(f).read_text())]
        print(format_ratings(group_ratings(results)))
        return 0
    if argv and argv[0] == "summary":
        # `summary DIR ...`: each DIR's run.log, the last 5 iterations
        rows = {Path(d).name: summarise((Path(d) / "run.log").read_text()) for d in argv[1:]}
        print(format_summary(rows))
        return 0
    if "--" in argv:
        split = argv.index("--")
        ours, rest = argv[:split], argv[split + 1 :]
    else:
        ours, rest = argv, []
    args = build_parser().parse_args(ours)
    settings = Settings(
        actor=args.pi_actor,
        student_weight=args.pi_student_weight,
        teacher_weight=args.pi_teacher_weight,
        beta_student=args.pi_beta_student,
        beta_teacher=args.pi_beta_teacher,
        teacher_clip=args.pi_teacher_clip,
        teacher_lr_scale=args.pi_teacher_lr_scale,
        placebo=args.pi_placebo,
        separate=args.pi_separate_teacher,
        critic_board=args.pi_critic_board,
        save_at=tuple(args.pi_save_at),
        save_dir=args.pi_save_dir,
    )
    from pokerlab.rl import global_arena, train

    distiller = install(settings, train, global_arena)
    print(f"pi-distill: {settings}", flush=True)
    runout = ["--no-critic-runout"] if settings.critic_board == "now" else ["--critic-runout"]
    sys.argv = ["poker-train", *rest, *runout, "--no-global-elo"]
    train.main()
    if args.pi_save_teacher is not None and distiller.teacher is not None:
        torch.save(distiller.teacher.state_dict(), args.pi_save_teacher)
    return 0


if __name__ == "__main__":
    sys.exit(main())

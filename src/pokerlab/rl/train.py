"""Self-play PPO training loop, and the `poker-train` entry point.

Requires the `rl` extra (`pip install -e ".[rl]"`).
"""

from __future__ import annotations

import argparse
import copy
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from pokerlab.engine.config import GameConfig
from pokerlab.engine.table import Table
from pokerlab.players.rl_agent import RLAgentPlayer
from pokerlab.players.scripted import get_bot_profile, list_bot_profiles
from pokerlab.rl.policy import PokerActorCritic, make_policy_fn
from pokerlab.rl.ppo import (
    PPOConfig,
    build_batch,
    build_model_from_checkpoint,
    load_checkpoint,
    ppo_update,
    save_checkpoint,
)
from pokerlab.rl.rollout import (
    Opponent,
    OpponentPool,
    SelfPlayCollector,
    policy_opponent,
    scripted_opponent,
)

DEFAULT_POOL_DIR = Path("checkpoints/pool")


def archived_opponents(
    directory: str | Path,
    game: GameConfig,
    *,
    device: str | torch.device = "cpu",
    limit: int = 5,
    on_skip: Callable[[Path, str], None] | None = None,
) -> list[Opponent]:
    """Seat previously trained agents, loaded from saved checkpoints.

    Without this the opponent pool resets to the scripted catalog on every run,
    and each new agent relearns from scratch against the same five bots; the
    snapshots that made the field interesting die with the process. Newest files
    first, on the assumption that a later agent is a stronger one.

    Unreadable or stale checkpoints are skipped rather than fatal: this scans a
    directory the user owns, and one bad file must not stop training.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []

    opponents: list[Opponent] = []
    for path in sorted(directory.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True):
        if len(opponents) >= limit:
            break
        try:
            model, _checkpoint = build_model_from_checkpoint(path, device=device)
        except Exception as exc:  # noqa: BLE001 - see docstring
            if on_skip is not None:
                on_skip(path, str(exc))
            continue
        opponents.append(policy_opponent(path.stem, make_policy_fn(model, device=device), game))
    return opponents


@dataclass(frozen=True)
class TrainConfig:
    hands_per_iteration: int = 256
    snapshot_every: int = 10
    max_snapshots: int = 5
    # The whole catalog, difficulty 1-5. Maniac in particular is the only
    # loose-aggressive preset: leave it out and the agent never has to learn to
    # defend against relentless aggression.
    bot_keys: tuple[str, ...] = ("random", "calling_station", "maniac", "rock", "shark")
    opponent_probability: float = 0.5
    gamma: float = 1.0
    lam: float = 0.95
    # None derives big_blind / starting_stack, which puts a stack-sized swing at
    # a value target of ~1.0. See SelfPlayCollector's reward_scale.
    reward_scale: float | None = None


class SelfPlayTrainer:
    def __init__(
        self,
        game_config: GameConfig,
        train_config: TrainConfig | None = None,
        ppo_config: PPOConfig | None = None,
        *,
        device: str | torch.device = "cpu",
        rng: random.Random | None = None,
        model: PokerActorCritic | None = None,
        extra_opponents: Sequence[Opponent] = (),
    ) -> None:
        self._game = game_config
        self._train = train_config if train_config is not None else TrainConfig()
        self._ppo = ppo_config if ppo_config is not None else PPOConfig()
        self._device = device
        self._rng = rng if rng is not None else random.Random()
        self._model = (model if model is not None else PokerActorCritic()).to(device)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._ppo.learning_rate)
        self._iteration = 0

        self._pool = OpponentPool(
            [*(scripted_opponent(key) for key in self._train.bot_keys), *extra_opponents],
            max_snapshots=self._train.max_snapshots,
        )
        self._reward_scale = (
            self._train.reward_scale
            if self._train.reward_scale is not None
            else game_config.big_blind / game_config.starting_stack
        )
        # The closure reads the live model, so updated weights take effect with
        # no rebuilding between iterations.
        self._collector = SelfPlayCollector(
            game_config,
            make_policy_fn(self._model, device=device),
            rng=self._rng,
            opponent_pool=self._pool,
            opponent_probability=self._train.opponent_probability,
            reward_scale=self._reward_scale,
        )

    @property
    def model(self) -> PokerActorCritic:
        return self._model

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    @property
    def iteration(self) -> int:
        return self._iteration

    def snapshot(self) -> None:
        """Freeze the current weights into the opponent pool."""
        frozen = copy.deepcopy(self._model).eval()
        for parameter in frozen.parameters():
            parameter.requires_grad_(False)
        self._pool.add_snapshot(
            policy_opponent(
                f"snap@{self._iteration}",
                make_policy_fn(frozen, device=self._device),
                self._game,
            )
        )

    def evaluate(self, bot_key: str, hands: int = 2000, *, seed: int | None = None) -> float:
        """The learner's win rate in bb/100 against a table of one catalog bot.

        Measured from stack deltas, not from collected trajectories: hands the
        learner wins without ever acting (everyone folds to its blind) produce
        no trajectory, and dropping them would bias the estimate downwards.

        **This number is very noisy.** Per-hand results are heavy-tailed -- a
        single hand can swing a whole stack -- so the mean converges far slower
        than 1/sqrt(n). Measured on one fixed model, the spread across seeds was
        ~570 bb/100 at 300 hands and still ~350 at 5000. Treat small differences
        between iterations as nothing at all; only a large, sustained move is
        signal. Proper variance reduction (duplicate deals: replay a seeded deck
        with rotated seats and average) is the fix, and is not built yet.
        """
        # Separate streams for dealing and for bot decisions: sharing one RNG
        # correlates the cards with the opponents' choices.
        # Separate streams for dealing and for bot decisions: sharing one RNG
        # correlates the cards with the opponents' choices.
        streams = random.Random(seed)
        rng = random.Random(streams.random())
        bot_rng = random.Random(streams.random())
        learner_seat = 0
        table = Table(
            self._game,
            self._seat_players(get_bot_profile(bot_key), learner_seat, bot_rng),
            rng=rng,
        )

        won = 0
        for _ in range(hands):
            before = table.stacks[learner_seat]
            table.play_hand()
            won += table.stacks[learner_seat] - before
            table.stacks = [self._game.starting_stack] * self._game.num_players
        return won / self._game.big_blind / hands * 100

    def _seat_players(self, profile, learner_seat: int, bot_rng: random.Random) -> list:
        return [
            RLAgentPlayer(
                f"s{seat}",
                "learner",
                policy_fn=make_policy_fn(self._model, device=self._device),
                big_blind=self._game.big_blind,
                starting_stack=self._game.starting_stack,
            )
            if seat == learner_seat
            else profile.factory(f"s{seat}", f"{profile.label}{seat}", bot_rng)
            for seat in range(self._game.num_players)
        ]

    def evaluate_duplicate(
        self, bot_key: str, deals: int = 400, *, seed: int | None = None
    ) -> float:
        """Win rate in bb/100 with the card luck cancelled out.

        Every deal is replayed once per seat, with the learner rotated into each
        one and the deck, the button and the opponents' random stream held
        fixed. Across the rotations the learner therefore holds *every* hand
        that was dealt, so what survives the average is decision quality rather
        than who got aces. This is the standard duplicate-poker trick, and it is
        cheap here only because `Deck` takes a `random.Random`.

        One deal costs `num_players` hands of compute but yields one hand of
        result, so `deals=400` at 6-max plays 2400 hands.

        **Measured: this did not help here.** At equal compute (1800 hands, six
        seeds, one fixed checkpoint) the spread was 259 bb/100 plain and 333
        duplicate. The cancellation relies on the hand playing out roughly the
        same whoever occupies a seat, and that fails when the learner's strategy
        differs wildly from the opponents' -- an agent shoving 45% of the time
        rewrites the hand rather than replaying it. Expect this to start paying
        off only once the agent plays a non-degenerate strategy.
        """
        profile = get_bot_profile(bot_key)
        num_players = self._game.num_players
        streams = random.Random(seed)
        total = 0.0

        for _ in range(deals):
            deal_seed = streams.randrange(2**31)
            bot_seed = streams.randrange(2**31)
            rotation_total = 0
            for learner_seat in range(num_players):
                players = self._seat_players(profile, learner_seat, random.Random(bot_seed))
                # A fresh Table starts the button at the lowest seat, so every
                # rotation plays the identical deal from the identical layout.
                table = Table(self._game, players, rng=random.Random(deal_seed))
                table.play_hand()
                rotation_total += table.stacks[learner_seat] - self._game.starting_stack
            total += rotation_total / num_players

        return total / self._game.big_blind / deals * 100

    def train_iteration(self) -> dict[str, float]:
        self._iteration += 1
        trajectories = self._collector.collect(
            self._train.hands_per_iteration, gamma=self._train.gamma, lam=self._train.lam
        )
        batch = build_batch(trajectories, device=self._device)
        stats = ppo_update(self._model, self._optimizer, batch, self._ppo)
        stats["iteration"] = float(self._iteration)
        stats["decisions"] = float(len(batch))
        stats["reward_bb"] = sum(t.reward for t in trajectories) / max(len(trajectories), 1)

        if self._train.snapshot_every and self._iteration % self._train.snapshot_every == 0:
            self.snapshot()
        return stats

    def train(
        self, iterations: int, *, checkpoint_path: str | Path | None = None
    ) -> list[dict[str, float]]:
        history = []
        for _ in range(iterations):
            stats = self.train_iteration()
            history.append(stats)
            if checkpoint_path is not None:
                save_checkpoint(
                    checkpoint_path,
                    self._model,
                    self._optimizer,
                    iteration=self._iteration,
                    metadata={"num_players": self._game.num_players},
                )
        return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a poker agent with self-play PPO.")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--hands", type=int, default=256, help="hands collected per iteration")
    parser.add_argument("--players", type=int, default=6)
    parser.add_argument("--stack", type=int, default=200)
    parser.add_argument("--sb", type=int, default=1)
    parser.add_argument("--bb", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/agent.pt"))
    parser.add_argument("--resume", action="store_true", help="load --checkpoint before training")
    parser.add_argument(
        "--pool-dir",
        type=Path,
        default=DEFAULT_POOL_DIR,
        help="archived agents are read from and written to here",
    )
    parser.add_argument(
        "--pool-models", type=int, default=5, help="how many archived agents to seat (0 disables)"
    )
    parser.add_argument(
        "--archive-every",
        type=int,
        default=25,
        help="iterations between permanent copies into --pool-dir (0 disables)",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=10, help="0 disables evaluation")
    parser.add_argument(
        "--eval-bot",
        default="shark",
        choices=[profile.key for profile in list_bot_profiles()],
    )
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    game = GameConfig(
        num_players=args.players, starting_stack=args.stack, small_blind=args.sb, big_blind=args.bb
    )
    archived = (
        archived_opponents(
            args.pool_dir,
            game,
            device=args.device,
            limit=args.pool_models,
            on_skip=lambda path, why: print(f"skipping {path.name}: {why}"),
        )
        if args.pool_models > 0
        else []
    )
    trainer = SelfPlayTrainer(
        game,
        TrainConfig(hands_per_iteration=args.hands),
        PPOConfig(learning_rate=args.lr),
        device=args.device,
        rng=random.Random(args.seed),
        extra_opponents=archived,
    )
    if args.resume:
        load_checkpoint(args.checkpoint, trainer.model, device=args.device)
        print(f"resumed from {args.checkpoint}")

    run_id = time.strftime("%Y%m%d-%H%M%S")
    print(f"device {args.device} | {args.players} seats | {args.hands} hands/iteration")
    if archived:
        print(f"seating {len(archived)} archived agents: {', '.join(o.label for o in archived)}")
    else:
        print(f"no archived agents in {args.pool_dir} -- scripted catalog only")
    for _ in range(args.iterations):
        stats = trainer.train_iteration()
        print(
            f"iter {int(stats['iteration']):4}  "
            f"reward {stats['reward_bb']:+7.2f} bb  "
            f"policy {stats['policy_loss']:+.4f}  "
            f"value {stats['value_loss']:8.3f}  "
            f"entropy {stats['entropy']:.3f}  "
            f"kl {stats['approx_kl']:.4f}  "
            f"clip {stats['clip_fraction']:.3f}"
        )
        iteration = int(stats["iteration"])
        if args.eval_every and iteration % args.eval_every == 0:
            win_rate = trainer.evaluate(args.eval_bot, seed=args.seed)
            print(f"        eval vs {args.eval_bot}: {win_rate:+.1f} bb/100")
            save_checkpoint(
                args.checkpoint, trainer.model, trainer.optimizer, iteration=trainer.iteration
            )
            print(f"        saved {args.checkpoint}")

        if args.archive_every and iteration % args.archive_every == 0:
            # A permanent copy, unlike --checkpoint which is overwritten every
            # time. These are what a later run seats as opponents.
            archive = Path(args.pool_dir) / f"agent-{run_id}-iter{iteration:05d}.pt"
            save_checkpoint(archive, trainer.model, iteration=iteration)
            print(f"        archived {archive}")


if __name__ == "__main__":
    main()

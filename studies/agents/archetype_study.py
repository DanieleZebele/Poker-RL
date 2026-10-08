"""Does a model read the HUD of opponents that really play differently, and does it pay?

    OMP_NUM_THREADS=1 PYTHONPATH=src:studies/agents python studies/agents/archetype_study.py --model '#1'
    ... --strength 2.0 --hands 40000 --decisions 15000 --jobs 16

`hud_study` shows a model made-up HUDs; here the HUD is real. The opponents are the model
itself (or `--opponent-model`) with a **fixed push on its logits** (`ARCHETYPE_BIASES`,
times `--strength`): a *nit* (more folds, fewer raises), a *maniaco* (more raises, fewer
folds), a *calling station* (more calls, fewer folds and raises). The push is added to the
network's own logits, so the cards still count -- a nit still plays aces -- and only the
style is bent. Their statistics come from a `StatsTracker` at the table like in a rated
session, so the HUD the hero reads is what those players actually did.

One seat is the hero (the model, unpushed); every other seat is the archetype. For each
archetype:

1. **The opponents' measured style**, next to the population's range: did the push make an
   archetype, and how far outside what the models have met?
2. **What the hero does with the HUD** (a counterfactual, as in `hud_study`): every state
   the hero decided in, shown again with the real HUD, with the population's median HUD
   (an opponent it cannot tell apart from the average) and with none. If the hero exploits,
   "vero" differs from "popolazione" in the direction the archetype calls for.
3. **What the HUD is worth in chips**: the same hands are played twice, from the same deck
   seeds, once with the hero reading the real HUD and once seeing the population's median
   HUD for everyone. The paired difference in bb/100 is what reading the opponents earns.
   Same cards, same seats; the hands diverge only where the hero's decisions do.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from hud_study import (
    BEHAVIOURS,
    Comparison,
    Population,
    action_group,
    argmax,
    evaluate_variants,
    hud_variant,
    population_from_members,
    profile_vector,
    total_variation,
    with_hud,
)

from pokerlab.engine.actions import Action, LegalAction
from pokerlab.engine.history import HandHistory
from pokerlab.engine.state import Street
from pokerlab.engine.stats import STATS, analyse_hand
from pokerlab.players.base import Observation, Player
from pokerlab.rl.action_space import ACTION_DIM, ALL_IN_BIN, CHECK_CALL_BIN, FOLD_BIN, RAISE_MIN_BIN

# ---- the archetypes: a push on the logits, by situation -------------------------

RAISES = tuple(range(RAISE_MIN_BIN, ALL_IN_BIN + 1))

# archetype -> situation -> {bin group: push in units of --strength}. Situations: preflop,
# "checked" (postflop, nothing to call) and "facing" (postflop, a bet to call).
ARCHETYPE_BIASES: dict[str, dict[str, dict[str, float]]] = {
    "nit": {
        "preflop": {"fold": 1.0, "raise": -1.0},
        "checked": {"raise": -1.0},
        "facing": {"fold": 1.0, "raise": -1.0},
    },
    "maniaco": {
        "preflop": {"fold": -1.0, "raise": 1.0},
        "checked": {"raise": 1.0},
        "facing": {"fold": -1.0, "raise": 1.0},
    },
    "calling station": {
        "preflop": {"fold": -1.0, "call": 1.0, "raise": -1.0},
        "checked": {"raise": -1.0},
        "facing": {"fold": -1.0, "call": 1.0, "raise": -1.0},
    },
}

# What an exploiting hero should do against each, for the behaviours of `hud_study`
# ("+" more often than against an average opponent, "-" less often).
EXPECTED: dict[str, dict[str, str]] = {
    "nit": {
        "preflop, nessun rilancio, ultimo 40%: rilancia (steal)": "+",
        "preflop, contro un rilancio, fascia 15-35%: fold": "+",
        "postflop, nessuna puntata, niente: punta (bluff)": "+",
        "postflop, contro una puntata, niente: fold": "+",
        "postflop, contro una puntata, coppia: fold": "+",
    },
    "maniaco": {
        "preflop, contro un rilancio, fascia 15-35%: fold": "-",
        "postflop, nessuna puntata, niente: punta (bluff)": "-",
        "postflop, nessuna puntata, mano forte: punta (value)": "+",
        "postflop, contro una puntata, niente: fold": "-",
        "postflop, contro una puntata, coppia: fold": "-",
    },
    "calling station": {
        "preflop, nessun rilancio, ultimo 40%: rilancia (steal)": "-",
        "postflop, nessuna puntata, niente: punta (bluff)": "-",
        "postflop, nessuna puntata, mano forte: punta (value)": "+",
        "postflop, contro una puntata, niente: fold": "+",
        "postflop, contro una puntata, coppia: fold": "+",
    },
}


def situation(observation: Observation) -> str:
    if observation.street == Street.PREFLOP:
        return "preflop"
    return "facing" if observation.current_bet_to_match > observation.my_current_bet else "checked"


def bias_vector(archetype: str, observation: Observation, strength: float) -> list[float]:
    """The push added to the logits of every bin, in this state."""
    push = ARCHETYPE_BIASES[archetype][situation(observation)]
    vector = [0.0] * ACTION_DIM
    for index in range(ACTION_DIM):
        group = "fold" if index == FOLD_BIN else "call" if index == CHECK_CALL_BIN else "raise"
        vector[index] = strength * push.get(group, 0.0)
    return vector


# ---- the players -----------------------------------------------------------------


class PushedAgent(Player):
    """A model with `bias_vector` added to its logits before sampling."""

    def __init__(self, player_id: str, name: str, model, archetype: str, strength: float, *,
                 big_blind: int, starting_stack: int) -> None:
        super().__init__(player_id, name)
        self.model, self.archetype, self.strength = model, archetype, strength
        self.big_blind, self.starting_stack = big_blind, starting_stack

    def act(self, observation: Observation, legal_actions: list[LegalAction]) -> Action:
        import torch

        from pokerlab.rl.action_space import action_index_to_action, legal_action_mask
        from pokerlab.rl.features import encode_observation

        mask = legal_action_mask(observation, legal_actions)
        features = encode_observation(observation, big_blind=self.big_blind,
                                      starting_stack=self.starting_stack, legal_mask=mask)
        with torch.no_grad():
            logits, _value = self.model(torch.tensor([features], dtype=torch.float32),
                                        torch.tensor([mask], dtype=torch.bool))
            logits = logits[0] + torch.tensor(bias_vector(self.archetype, observation, self.strength))
            index = int(torch.distributions.Categorical(logits=logits).sample().item())
        return action_index_to_action(index, observation, legal_actions)


class Hero(Player):
    """The model, reading either the real HUD or `hud` (the same for every opponent), and
    keeping the states it decided in (with the real HUD) when asked to."""

    def __init__(self, inner: Player, hud: tuple[float, ...] | None, real: bool,
                 kept: list[tuple[Observation, list[LegalAction]]] | None) -> None:
        super().__init__(inner.player_id, inner.name)
        self.inner, self.hud, self.real, self.kept = inner, hud, real, kept

    def act(self, observation: Observation, legal_actions: list[LegalAction]) -> Action:
        if self.kept is not None:
            self.kept.append((observation, list(legal_actions)))
        seen = observation if self.real else with_hud(observation, self.hud)
        return self.inner.act(seen, legal_actions)


# ---- one worker's share ------------------------------------------------------------


@dataclass(frozen=True)
class ArchetypeJob:
    seed: int
    hands: int
    hero_path: str
    opponent_path: str
    archetype: str
    strength: float
    real_hud: bool  # False: the hero sees `population_hud` for every opponent
    population_hud: tuple[float, ...]
    keep: int  # states of the hero to keep (0: none)
    weights: tuple[float, ...]
    stack_min_bb: float
    stack_max_bb: float
    small_blind: int
    big_blind: int
    session_hands: int


@dataclass
class ArchetypeRun:
    hero_bb: list[float] = field(default_factory=list)  # one per hand, in deal order
    opponent_events: list[int] = field(default_factory=lambda: [0] * len(STATS))
    opponent_chances: list[int] = field(default_factory=lambda: [0] * len(STATS))
    states: list[tuple[Observation, list[LegalAction]]] = field(default_factory=list)


def run_job(job: ArchetypeJob) -> ArchetypeRun:
    """`job.hands` hands, one hero seat a table, every other seat the archetype. Both arms
    of a pair are dealt the same: the table's rng only shuffles and draws the table, the
    stacks, the table size and the hero's seat have their own seeded rngs, and torch is
    reseeded."""
    import torch

    from pokerlab.players.rl_agent import RLAgentPlayer
    from pokerlab.rl.policy import make_policy_fn
    from pokerlab.rl.ppo import build_model_from_checkpoint
    from pokerlab.rl.rollout import TableBank
    from pokerlab.rl.table_mix import TableMix

    torch.set_num_threads(1)
    torch.manual_seed(job.seed)
    hero_model, _ = build_model_from_checkpoint(job.hero_path)
    opponent_model = hero_model if job.opponent_path == job.hero_path else build_model_from_checkpoint(job.opponent_path)[0]
    policy = make_policy_fn(hero_model)
    mix = TableMix(weights=job.weights, stack_min_bb=job.stack_min_bb, stack_max_bb=job.stack_max_bb,
                   small_blind=job.small_blind, big_blind=job.big_blind)
    bank = TableBank(mix, random.Random(job.seed), stack_rng=random.Random(job.seed + 1))
    seat_rng = random.Random(job.seed + 2)
    size_rng = random.Random(job.seed + 3)
    run = ArchetypeRun()
    kept: list[tuple[Observation, list[LegalAction]]] | None = [] if job.keep else None

    def on_hand(hand: HandHistory, hero_seat: int) -> None:
        run.hero_bb.append((hand.final_stacks[hero_seat] - hand.starting_stacks[hero_seat]) / hand.big_blind)
        dealt = sorted(hand.starting_stacks)
        counts = analyse_hand(hand.actions, dealt=dealt, button_seat=hand.button_seat,
                              board_cards=len(hand.community_cards), player_ids={s: f"s{s}" for s in dealt})
        for seat in dealt:
            if seat == hero_seat:
                continue
            for slot, (events, chances) in enumerate(zip(counts[f"s{seat}"].events, counts[f"s{seat}"].opportunities)):
                run.opponent_events[slot] += events
                run.opponent_chances[slot] += chances

    while len(run.hero_bb) < job.hands:
        size = mix.draw_size(size_rng)
        hero_seat = seat_rng.randrange(size)
        for seat, proxy in enumerate(bank.seats(size)):
            if seat == hero_seat:
                inner = RLAgentPlayer(proxy.player_id, proxy.name, policy_fn=policy,
                                      big_blind=mix.big_blind, starting_stack=mix.starting_stack)
                proxy.inner = Hero(inner, job.population_hud, job.real_hud, kept)
            else:
                proxy.inner = PushedAgent(proxy.player_id, proxy.name, opponent_model, job.archetype, job.strength,
                                          big_blind=mix.big_blind, starting_stack=mix.starting_stack)
        hands = min(job.session_hands, job.hands - len(run.hero_bb))
        bank.play_session(size, hands, on_hand=lambda hand, seat=hero_seat: on_hand(hand, seat))
    if kept is not None:
        step = max(1, len(kept) // job.keep)
        run.states = kept[::step][: job.keep]
    return run


# ---- the report ---------------------------------------------------------------------


def paired_difference(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Mean of `a - b` hand by hand, in bb/100, and its 95% half-interval."""
    differences = [x - y for x, y in zip(a, b)]
    n = len(differences)
    mean = sum(differences) / n
    variance = sum((d - mean) ** 2 for d in differences) / max(n - 1, 1)
    return 100 * mean, 100 * 1.96 * math.sqrt(variance / n)


def _pct(value: float | None, width: int = 8) -> str:
    return f"{100 * value:{width - 1}.1f}%" if value is not None else f"{'-':>{width}}"


def format_style(events: Sequence[int], chances: Sequence[int], population: Population) -> list[str]:
    short = {"vpip": "VPIP", "pfr": "PFR", "three_bet": "3bet", "fold_to_three_bet": "f3bet", "steal": "steal",
             "aggression": "aggr", "cbet": "cbet", "fold_to_cbet": "fcbet", "wtsd": "WTSD"}
    lines = [f"    {'':<24}" + "".join(f"{short[n]:>7}" for n in STATS)]
    lines.append(f"    {'popolazione: minimo':<24}" + "".join(f"{100 * population.low[n]:6.0f}%" for n in STATS))
    lines.append(f"    {'popolazione: massimo':<24}" + "".join(f"{100 * population.high[n]:6.0f}%" for n in STATS))
    cells = []
    for index, name in enumerate(STATS):
        if not chances[index]:
            cells.append(f"{'-':>7}")
            continue
        rate = events[index] / chances[index]
        outside = not population.low[name] <= rate <= population.high[name]
        cells.append(f"{100 * rate:5.0f}{'!' if outside else ' '}%")
    lines.append(f"    {'misurato su di loro':<24}" + "".join(cells))
    return lines


def format_counterfactual(comparison: Comparison, archetype: str, *, min_count: int) -> list[str]:
    real = comparison.distributions["vero"]
    lines = []
    for other in ("popolazione", "nessuno"):
        rows = comparison.distributions[other]
        tv = sum(total_variation(p, q) for p, q in zip(real, rows)) / len(real)
        changed = sum(argmax(p) != argmax(q) for p, q in zip(real, rows)) / len(real)
        lines.append(f"    HUD vero contro HUD {other:<12} TV {100 * tv:5.1f}%   azione piu' probabile diversa {100 * changed:5.1f}%")
    lines += ["", f"    {'comportamento':<55} {'n':>6} {'vero':>8} {'popolaz.':>8} {'nessuno':>8} {'vero-pop':>9}  atteso"]
    expected = EXPECTED.get(archetype, {})
    for label, picks, groups in BEHAVIOURS:
        chosen = [i for i, c in enumerate(comparison.contexts) if picks(c)]
        if len(chosen) < min_count:
            lines.append(f"    {label:<55} {len(chosen):>6}  (troppo poche)")
            continue
        shares = {}
        for name in ("vero", "popolazione", "nessuno"):
            rows = comparison.distributions[name]
            shares[name] = sum(sum(rows[i][b] for b in range(ACTION_DIM) if action_group(b) in groups)
                               for i in chosen) / len(chosen)
        delta = shares["vero"] - shares["popolazione"]
        want = expected.get(label, "")
        verdict = ""
        if want:
            verdict = "si'" if (delta > 0.005 and want == "+") or (delta < -0.005 and want == "-") else "no"
        lines.append(f"    {label:<55} {len(chosen):>6}" + "".join(_pct(shares[n]) for n in ("vero", "popolazione", "nessuno"))
                     + f" {100 * delta:+8.1f}  {want:>2} {verdict}")
    lines.append("    atteso: + / - = cosa farebbe chi sfrutta questo avversario rispetto a uno medio;"
                 " si'/no = se la differenza (vero-pop, oltre 0,5 punti) va in quella direzione")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    from duel_power import resolve_model

    from pokerlab.rl.table_mix import (
        DEFAULT_BIG_BLIND,
        DEFAULT_SMALL_BLIND,
        DEFAULT_STACK_MAX_BB,
        DEFAULT_STACK_MIN_BB,
        DEFAULT_TABLE_WEIGHTS,
        SIZES,
        TableMix,
    )

    parser = argparse.ArgumentParser(description="l'eroe contro avversari-archetipo con una spinta sui logit")
    parser.add_argument("--model", default="#1", help="l'eroe: etichetta, percorso o #N")
    parser.add_argument("--opponent-model", default=None, help="il modello spinto (default: lo stesso dell'eroe)")
    parser.add_argument("--archetypes", nargs="+", default=list(ARCHETYPE_BIASES), choices=list(ARCHETYPE_BIASES))
    parser.add_argument("--strength", type=float, default=2.0, help="spinta sui logit (2 = probabilita' x7 circa)")
    parser.add_argument("--hands", type=int, default=40000, help="mani per archetipo, giocate due volte")
    parser.add_argument("--decisions", type=int, default=15000, help="stati dell'eroe per il controfattuale")
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--players", type=int, default=None)
    parser.add_argument("--stack-bb", type=float, default=None)
    parser.add_argument("--session-hands", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--root", type=Path, default=Path("checkpoints"), help="la cartella dei checkpoint (per cercare un modello per etichetta)")
    parser.add_argument("--global-dir", type=Path, default=Path("checkpoints/global"))
    parser.add_argument("--min-count", type=int, default=50)
    args = parser.parse_args(argv)

    hero_label, hero_path, rating = resolve_model(args.model, global_dir=args.global_dir, root=args.root)
    opponent_path = hero_path
    opponent_label = "lo stesso modello"
    if args.opponent_model:
        opponent_label, opponent_path, _ = resolve_model(args.opponent_model, global_dir=args.global_dir, root=args.root)
    population = population_from_members(json.loads((args.global_dir / "registry.json").read_text())["members"])
    population_hud = profile_vector(population.median, population.per_hand)

    weights = list(DEFAULT_TABLE_WEIGHTS)
    if args.players is not None:
        weights = [1.0 if size == args.players else 0.0 for size in SIZES]
    low, high = (args.stack_bb, args.stack_bb) if args.stack_bb else (DEFAULT_STACK_MIN_BB, DEFAULT_STACK_MAX_BB)
    mix = TableMix(weights=tuple(weights), stack_min_bb=low, stack_max_bb=high,
                   small_blind=DEFAULT_SMALL_BLIND, big_blind=DEFAULT_BIG_BLIND)
    workers = max(1, min(args.jobs, args.hands))
    shares = [args.hands // workers + (1 if i < args.hands % workers else 0) for i in range(workers)]

    print(f"=== {hero_label}" + (f" (rating {rating:.0f})" if rating is not None else "")
          + f" contro archetipi ({opponent_label} con una spinta di {args.strength:g} sui logit) ===")
    print(f"{args.hands:,} mani per archetipo, giocate due volte dagli stessi mazzi (HUD vero / HUD della popolazione)")
    rng = random.Random(args.seed)
    for archetype in args.archetypes:
        seeds = [rng.randrange(2**30) for _ in range(workers)]
        runs: dict[bool, list[ArchetypeRun]] = {}
        for real in (True, False):
            jobs = [
                ArchetypeJob(seed, share, str(hero_path), str(opponent_path), archetype, args.strength, real,
                             population_hud, (args.decisions // workers + 1) if real else 0, tuple(weights), low, high,
                             mix.small_blind, mix.big_blind, args.session_hands)
                for seed, share in zip(seeds, shares)
            ]
            print(f"  {archetype}: gioco con HUD {'vero' if real else 'della popolazione'}...", file=sys.stderr, flush=True)
            with ProcessPoolExecutor(max_workers=workers) as executor:
                runs[real] = list(executor.map(run_job, jobs))

        real_runs, blind_runs = runs[True], runs[False]
        events = [sum(r.opponent_events[i] for r in real_runs) for i in range(len(STATS))]
        chances = [sum(r.opponent_chances[i] for r in real_runs) for i in range(len(STATS))]
        states = [state for r in real_runs for state in r.states]
        comparison = evaluate_variants(
            hero_path, states,
            {"vero": lambda observation: observation, "popolazione": hud_variant(population_hud), "nessuno": hud_variant(None)},
            big_blind=mix.big_blind, starting_stack=mix.starting_stack,
        )
        real_bb = [x for r in real_runs for x in r.hero_bb]
        blind_bb = [x for r in blind_runs for x in r.hero_bb]
        difference, half = paired_difference(real_bb, blind_bb)

        print(f"\n== {archetype.upper()} " + "=" * max(0, 72 - len(archetype)))
        print("  lo stile degli avversari spinti (! = fuori dall'intervallo della popolazione):")
        for line in format_style(events, chances, population):
            print(line)
        print(f"\n  cosa fa l'eroe con l'HUD ({len(states):,} sue decisioni riproposte):")
        for line in format_counterfactual(comparison, archetype, min_count=args.min_count):
            print(line)
        print("\n  quanto rende leggere l'HUD (stesse mani, stessi mazzi):")
        print(f"    eroe con HUD vero         {100 * sum(real_bb) / len(real_bb):+8.1f} bb/100")
        print(f"    eroe con HUD popolazione  {100 * sum(blind_bb) / len(blind_bb):+8.1f} bb/100")
        print(f"    differenza (vero - pop)   {difference:+8.1f} +- {half:.1f} bb/100 (95%)")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Does a model play differently against opponents with different HUDs?

    OMP_NUM_THREADS=1 PYTHONPATH=src:studies/agents python studies/agents/hud_study.py --model '#1'
    ... --model <label | path/to/agent.pt | #N> --decisions 20000 --jobs 16

A counterfactual: the model plays against itself (as in `agent_study`), every state it
decides in is kept, and then each state is shown to it again **identical except for the
opponents' statistics** (`Observation.seat_stats`, the HUD of `engine/stats.py`), once per
HUD profile, every opponent at the table given the same one. Whatever moves in its action
distribution was moved by the HUD and by nothing else: same cards, same board, same pot,
same history. The distributions are compared exactly (the softmax over the 11 bins), not
by sampling, so a small change is visible.

The profiles:

- **sconosciuto**: no statistics, what a model sees on the first hand of a table;
- **popolazione**: the median of every rated model's measured style (the reference every
  other profile is compared with);
- **piu' tight / piu' loose della popolazione**: the real measured styles of the rated
  models with the lowest and the highest VPIP -- opponents the models do meet;
- **nit, TAG, LAG, maniaco, calling station**: human archetypes. The models never trained
  against anyone like that (the population's WTSD is never below ~0.7), so a profile with
  a rate outside the population's range is marked "fuori distribuzione": what the model
  does there is an extrapolation, not a learned answer.

A profile is turned into the exact 20 numbers a `StatsTracker` would give a player seen
for `WINDOW` (200) hands with those rates, with the opportunities per hand of the
population (a nit has fewer flops to show down than a maniac in reality; here only the
rates move, which is the cleaner experiment).

The report: how far each profile moves the distribution from the reference (total
variation, the share of decisions whose most likely action changes), by street; the
probability mass of fold / check-call / small raise / big raise / all-in per profile and
situation; and a few behaviours the HUD should move if the model reads it (stealing with
weak hands, bluffing with nothing, folding to a bet, raising for value).
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

from agent_study import STRONG_MADE, STRONG_TIERS, TIER_OF, TIERS, hand_class, made_class

from pokerlab.engine.actions import Action, ActionType, LegalAction
from pokerlab.engine.state import Street
from pokerlab.engine.stats import STATS, WINDOW, HandCounts, StatsTracker
from pokerlab.players.base import Observation, Player
from pokerlab.rl.action_space import (
    ACTION_DIM,
    ALL_IN_BIN,
    CHECK_CALL_BIN,
    FOLD_BIN,
    POT_FRACTIONS,
    RAISE_MIN_BIN,
)

# ---- the profiles --------------------------------------------------------------

# Human archetypes, rate per statistic (`engine/stats.py` names).
ARCHETYPES: dict[str, dict[str, float]] = {
    "nit": {"vpip": 0.12, "pfr": 0.09, "three_bet": 0.03, "fold_to_three_bet": 0.75, "steal": 0.15,
            "aggression": 0.35, "cbet": 0.50, "fold_to_cbet": 0.60, "wtsd": 0.20},
    "TAG": {"vpip": 0.22, "pfr": 0.18, "three_bet": 0.08, "fold_to_three_bet": 0.55, "steal": 0.35,
            "aggression": 0.50, "cbet": 0.65, "fold_to_cbet": 0.45, "wtsd": 0.27},
    "LAG": {"vpip": 0.35, "pfr": 0.28, "three_bet": 0.14, "fold_to_three_bet": 0.40, "steal": 0.50,
            "aggression": 0.60, "cbet": 0.75, "fold_to_cbet": 0.35, "wtsd": 0.30},
    "maniaco": {"vpip": 0.70, "pfr": 0.50, "three_bet": 0.30, "fold_to_three_bet": 0.20, "steal": 0.80,
                "aggression": 0.80, "cbet": 0.90, "fold_to_cbet": 0.20, "wtsd": 0.40},
    "calling station": {"vpip": 0.60, "pfr": 0.06, "three_bet": 0.02, "fold_to_three_bet": 0.30,
                        "steal": 0.10, "aggression": 0.15, "cbet": 0.30, "fold_to_cbet": 0.15, "wtsd": 0.55},
}
UNKNOWN = "sconosciuto"
REFERENCE = "popolazione"
MIN_STYLE_HANDS = 5000  # a member's style over fewer seat-hands is too noisy to stand for a profile


@dataclass(frozen=True)
class Profile:
    name: str
    rates: dict[str, float] | None  # None: no statistics at all
    out_of_range: tuple[str, ...] = ()  # statistics outside the population's measured range


@dataclass(frozen=True)
class Population:
    """The rated models' measured styles: per statistic the median rate, the range of
    rates, and how many opportunities a hand gives (`style` in each member)."""

    median: dict[str, float]
    low: dict[str, float]
    high: dict[str, float]
    per_hand: dict[str, float]
    tightest: dict[str, float]
    loosest: dict[str, float]
    members: int


def population_from_members(members: Sequence[Mapping]) -> Population:
    styled = [m for m in members if m.get("style") and m.get("style_hands", 0) >= MIN_STYLE_HANDS]
    if not styled:
        raise SystemExit("nessun modello nel registro ha uno stile misurato")

    def rates(member: Mapping) -> dict[str, float]:
        return {name: (e / o if o else 0.0) for name, (e, o) in member["style"].items() if name in STATS}

    all_rates = [rates(m) for m in styled]
    median = {name: statistics.median(r[name] for r in all_rates) for name in STATS}
    low = {name: min(r[name] for r in all_rates) for name in STATS}
    high = {name: max(r[name] for r in all_rates) for name in STATS}
    per_hand = {
        name: statistics.median(m["style"][name][1] / m["style_hands"] for m in styled) for name in STATS
    }
    by_vpip = sorted(all_rates, key=lambda r: r["vpip"])
    return Population(median, low, high, per_hand, by_vpip[0], by_vpip[-1], len(styled))


def profiles_for(population: Population) -> list[Profile]:
    def outside(rates: Mapping[str, float]) -> tuple[str, ...]:
        return tuple(n for n in STATS if not population.low[n] - 1e-9 <= rates[n] <= population.high[n] + 1e-9)

    found = [
        Profile(UNKNOWN, None),
        Profile(REFERENCE, population.median),
        Profile("piu' tight della popolazione", population.tightest),
        Profile("piu' loose della popolazione", population.loosest),
    ]
    found += [Profile(name, rates, outside(rates)) for name, rates in ARCHETYPES.items()]
    return found


def profile_vector(rates: Mapping[str, float], per_hand: Mapping[str, float], hands: int = WINDOW) -> tuple[float, ...]:
    """The HUD a `StatsTracker` gives a player seen for `hands` hands at these rates, with
    the population's opportunities per hand. Built through the tracker itself, so the
    numbers are exactly the ones the model reads at a table."""
    tracker = StatsTracker(window=hands)
    opportunities = [round(per_hand[name] * hands) for name in STATS]
    events = [round(rates[name] * chances) for name, chances in zip(STATS, opportunities)]
    # The window only sums its hands: the counts can all ride on the first one.
    tracker.add("p", HandCounts(tuple(events), tuple(opportunities)))
    for _ in range(hands - 1):
        tracker.add("p", HandCounts((0,) * len(STATS), (0,) * len(STATS)))
    vector = tracker.vector("p")
    assert vector is not None
    # The tracker can only hold whole events, which on a statistic with few chances a
    # window (a c-bet: ~4 in 200 hands) would round a rate to quarters: put the exact rate
    # in its slot (after the "seen" flag and the hands, a rate and a count per statistic).
    slots = list(vector)
    for index, name in enumerate(STATS):
        slots[2 + 2 * index] = rates[name]
    return tuple(slots)


def with_hud(observation: Observation, vector: tuple[float, ...] | None) -> Observation:
    """The same state with every opponent given `vector` (None: no statistics at all)."""
    if vector is None:
        return replace(observation, seat_stats={})
    others = {seat.seat: vector for seat in observation.seats if seat.seat != observation.my_seat}
    return replace(observation, seat_stats=others)


# ---- what a state is ------------------------------------------------------------

GROUPS = ("fold", "check/call", "raise piccolo", "raise grande", "all-in")
_BIG_FROM = 3 + POT_FRACTIONS.index(0.75)  # bins 75% of the pot and up


def action_group(index: int) -> str:
    if index == FOLD_BIN:
        return "fold"
    if index == CHECK_CALL_BIN:
        return "check/call"
    if index == ALL_IN_BIN:
        return "all-in"
    return "raise grande" if index >= _BIG_FROM else "raise piccolo"


assert RAISE_MIN_BIN < _BIG_FROM < ALL_IN_BIN


def grouped(probabilities: Sequence[float]) -> dict[str, float]:
    mass = dict.fromkeys(GROUPS, 0.0)
    for index, p in enumerate(probabilities):
        mass[action_group(index)] += p
    return mass


def raises_this_street(observation: Observation, big_blind: int) -> int:
    """Bets and raises already made on the street being played (blinds not counted; an
    all-in for no more than the bet is a call)."""
    level = big_blind if observation.street == Street.PREFLOP else 0
    raises = 0
    for record in observation.action_history:
        if record.street != observation.street or record.action_type == ActionType.POST_BLIND:
            continue
        aggressive = record.action_type in (ActionType.BET, ActionType.RAISE) or (
            record.action_type == ActionType.ALL_IN and record.amount > level
        )
        if aggressive:
            raises += 1
            level = record.amount
    return raises


@dataclass(frozen=True)
class Context:
    street: str
    facing: bool  # something to call
    raises: int
    tier: str
    made: str | None  # postflop only


def context_of(observation: Observation, big_blind: int) -> Context:
    postflop = observation.street != Street.PREFLOP
    return Context(
        street=observation.street.value,
        facing=observation.current_bet_to_match > observation.my_current_bet,
        raises=raises_this_street(observation, big_blind),
        tier=TIER_OF[hand_class(observation.hole_cards)],
        made=made_class(observation.hole_cards, observation.community_cards) if postflop else None,
    )


# The behaviours a HUD should move, each: (label, which states, which action groups).
RAISE = ("raise piccolo", "raise grande", "all-in")
BEHAVIOURS: tuple[tuple[str, Callable[[Context], bool], tuple[str, ...]], ...] = (
    ("preflop, nessun rilancio, ultimo 40%: rilancia (steal)",
     lambda c: c.street == "preflop" and c.raises == 0 and c.tier == TIERS[-1], RAISE),
    ("preflop, nessun rilancio, fascia 15-35%: rilancia",
     lambda c: c.street == "preflop" and c.raises == 0 and c.tier == TIERS[2], RAISE),
    ("preflop, contro un rilancio, fascia 15-35%: fold",
     lambda c: c.street == "preflop" and c.raises == 1 and c.tier == TIERS[2], ("fold",)),
    ("preflop, contro un rilancio, top 15%: rilancia",
     lambda c: c.street == "preflop" and c.raises == 1 and c.tier in STRONG_TIERS, RAISE),
    ("postflop, nessuna puntata, niente: punta (bluff)",
     lambda c: c.made == "nulla" and not c.facing, RAISE),
    ("postflop, nessuna puntata, mano forte: punta (value)",
     lambda c: c.made in STRONG_MADE and not c.facing, RAISE),
    ("postflop, contro una puntata, niente: fold",
     lambda c: c.made == "nulla" and c.facing, ("fold",)),
    ("postflop, contro una puntata, coppia: fold",
     lambda c: c.made == "coppia" and c.facing, ("fold",)),
    ("postflop, contro una puntata, top pair+: rilancia",
     lambda c: c.made == "top pair+" and c.facing, RAISE),
    ("tutte le decisioni: all-in", lambda c: True, ("all-in",)),
)


# ---- the comparison --------------------------------------------------------------


def total_variation(p: Sequence[float], q: Sequence[float]) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


def argmax(values: Sequence[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


@dataclass
class Comparison:
    """Every decision's action distribution under every profile, and its context."""

    contexts: list[Context]
    distributions: dict[str, list[list[float]]]  # profile -> one distribution per decision


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _pct(value: float | None, width: int = 6) -> str:
    return f"{100 * value:{width - 1}.1f}%" if value is not None else f"{'-':>{width}}"


def format_profiles(profiles: Sequence[Profile], population: Population) -> list[str]:
    short = {"vpip": "VPIP", "pfr": "PFR", "three_bet": "3bet", "fold_to_three_bet": "f3bet", "steal": "steal",
             "aggression": "aggr", "cbet": "cbet", "fold_to_cbet": "fcbet", "wtsd": "WTSD"}
    lines = [f"  {'profilo':<30}" + "".join(f"{short[n]:>7}" for n in STATS)]
    lines.append(f"  {'(popolazione: minimo)':<30}" + "".join(f"{100 * population.low[n]:6.0f}%" for n in STATS))
    lines.append(f"  {'(popolazione: massimo)':<30}" + "".join(f"{100 * population.high[n]:6.0f}%" for n in STATS))
    for profile in profiles:
        if profile.rates is None:
            lines.append(f"  {profile.name:<30}" + "  (nessuna statistica)")
            continue
        cells = "".join(
            f"{100 * profile.rates[n]:5.0f}{'!' if n in profile.out_of_range else ' '}%" for n in STATS
        )
        lines.append(f"  {profile.name:<30}{cells}")
    lines.append(f"  ! = fuori dall'intervallo misurato sui {population.members} modelli: la risposta e' un'estrapolazione")
    return lines


def format_shift(comparison: Comparison, profiles: Sequence[Profile]) -> list[str]:
    reference = comparison.distributions[REFERENCE]
    streets = ("preflop", "flop", "turn", "river")
    lines = [
        "  quanto si sposta la distribuzione delle azioni rispetto a 'popolazione':",
        "  TV = distanza di variazione totale media (0 = identica, 100% = disgiunta);",
        "  cambia = quota di decisioni in cui l'azione piu' probabile e' un'altra",
        "",
        f"  {'profilo':<30} {'TV':>7} {'cambia':>7}   " + "".join(f"{('TV ' + s):>11}" for s in streets),
    ]
    for profile in profiles:
        if profile.name == REFERENCE:
            continue
        rows = comparison.distributions[profile.name]
        distances = [total_variation(p, q) for p, q in zip(rows, reference)]
        changed = [argmax(p) != argmax(q) for p, q in zip(rows, reference)]
        by_street = []
        for street in streets:
            picked = [d for d, c in zip(distances, comparison.contexts) if c.street == street]
            by_street.append(_pct(_mean(picked), 11))
        lines.append(
            f"  {profile.name:<30} {_pct(_mean(distances), 7)} {_pct(_mean(changed), 7)}   " + "".join(by_street)
        )
    return lines


def format_mass(comparison: Comparison, profiles: Sequence[Profile], *, min_count: int) -> list[str]:
    situations = (
        ("preflop, nessun rilancio", lambda c: c.street == "preflop" and c.raises == 0),
        ("preflop, contro un rilancio", lambda c: c.street == "preflop" and c.raises >= 1),
        ("postflop, nessuna puntata", lambda c: c.street != "preflop" and not c.facing),
        ("postflop, contro una puntata", lambda c: c.street != "preflop" and c.facing),
    )
    lines = []
    for title, picks in situations:
        chosen = [i for i, c in enumerate(comparison.contexts) if picks(c)]
        lines += ["", f"  {title} ({len(chosen):,} decisioni):",
                  f"    {'profilo':<30}" + "".join(f"{g:>14}" for g in GROUPS)]
        if len(chosen) < min_count:
            lines.append("    (troppo poche)")
            continue
        for profile in profiles:
            rows = comparison.distributions[profile.name]
            mass = defaultdict(float)
            for i in chosen:
                for group, value in grouped(rows[i]).items():
                    mass[group] += value
            lines.append(f"    {profile.name:<30}" + "".join(_pct(mass[g] / len(chosen), 14) for g in GROUPS))
    return lines


def format_behaviours(comparison: Comparison, profiles: Sequence[Profile], *, min_count: int) -> list[str]:
    names = [p.name for p in profiles]
    width = 11
    header = "".join(f"{_short(name):>{width}}" for name in names)
    lines = [f"  {'comportamento':<55} {'n':>6}{header}"]
    for label, picks, groups in BEHAVIOURS:
        chosen = [i for i, c in enumerate(comparison.contexts) if picks(c)]
        if len(chosen) < min_count:
            lines.append(f"  {label:<55} {len(chosen):>6}  (troppo poche)")
            continue
        cells = []
        for name in names:
            rows = comparison.distributions[name]
            share = sum(sum(rows[i][b] for b in range(ACTION_DIM) if action_group(b) in groups) for i in chosen)
            cells.append(_pct(share / len(chosen), width))
        lines.append(f"  {label:<55} {len(chosen):>6}" + "".join(cells))
    lines += ["", "  colonne: " + ", ".join(f"{_short(n)} = {n}" for n in names if _short(n) != n)]
    lines += [
        "  cosa aspettarsi da un modello che legge l'HUD: piu' steal contro chi folda molto (nit),",
        "  meno bluff e piu' puntate di valore contro chi chiama tutto (calling station), fold piu'",
        "  rari contro chi punta molto (maniaco). Valori uguali in ogni colonna = l'HUD non conta.",
    ]
    return lines


def _short(name: str) -> str:
    return {
        "piu' tight della popolazione": "pop. tight",
        "piu' loose della popolazione": "pop. loose",
        "calling station": "station",
        UNKNOWN: "sconosc.",
        REFERENCE: "popolaz.",
    }.get(name, name)


def format_report(comparison: Comparison, profiles: Sequence[Profile], population: Population, *,
                  label: str, min_count: int) -> list[str]:
    def section(title: str) -> list[str]:
        return ["", f"== {title} " + "=" * max(0, 74 - len(title))]

    intro = (
        f"{len(comparison.contexts):,} decisioni prese contro se stesso, riproposte identiche con "
        "l'HUD di ogni avversario sostituito"
    )
    lines = [f"=== come cambia {label} con l'HUD degli avversari ===", intro]
    lines += section("PROFILI (frequenze per statistica)") + format_profiles(profiles, population)
    lines += section("QUANTO CAMBIANO LE AZIONI") + format_shift(comparison, profiles)
    lines += section("COMPORTAMENTI (probabilita' media dell'azione)")
    lines += format_behaviours(comparison, profiles, min_count=min_count)
    lines += section("DISTRIBUZIONE DELLE AZIONI PER SITUAZIONE") + format_mass(comparison, profiles, min_count=min_count)
    return lines


# ---- playing and evaluating (torch, imported lazily) ----------------------------


class _Recorder(Player):
    """Plays through `inner` and keeps every state it was asked to decide in."""

    def __init__(self, inner: Player, kept: list[tuple[Observation, list[LegalAction]]]) -> None:
        super().__init__(inner.player_id, inner.name)
        self.inner = inner
        self.kept = kept

    def act(self, observation: Observation, legal_actions: list[LegalAction]) -> Action:
        self.kept.append((observation, list(legal_actions)))
        return self.inner.act(observation, legal_actions)


@dataclass(frozen=True)
class CollectJob:
    seed: int
    decisions: int
    path: str
    weights: tuple[float, ...]
    stack_min_bb: float
    stack_max_bb: float
    small_blind: int
    big_blind: int
    session_hands: int


def collect_job(job: CollectJob) -> list[tuple[Observation, list[LegalAction]]]:
    """Self-play until `job.decisions` states are kept, a table drawn per session."""
    import torch

    from pokerlab.players.rl_agent import RLAgentPlayer
    from pokerlab.rl.policy import make_policy_fn
    from pokerlab.rl.ppo import build_model_from_checkpoint
    from pokerlab.rl.rollout import TableBank
    from pokerlab.rl.table_mix import TableMix

    torch.set_num_threads(1)
    torch.manual_seed(job.seed)
    model, _checkpoint = build_model_from_checkpoint(job.path)
    policy = make_policy_fn(model)
    mix = TableMix(weights=job.weights, stack_min_bb=job.stack_min_bb, stack_max_bb=job.stack_max_bb,
                   small_blind=job.small_blind, big_blind=job.big_blind)
    rng = random.Random(job.seed)
    bank = TableBank(mix, rng)
    kept: list[tuple[Observation, list[LegalAction]]] = []
    while len(kept) < job.decisions:
        size = mix.draw_size(rng)
        for proxy in bank.seats(size):
            proxy.inner = _Recorder(
                RLAgentPlayer(proxy.player_id, proxy.name, policy_fn=policy,
                              big_blind=mix.big_blind, starting_stack=mix.starting_stack),
                kept,
            )
        # Short sessions keep tables varied; the statistics fill in as in a rated session.
        bank.play_session(size, job.session_hands)
    return kept[: job.decisions]


def evaluate_variants(
    path: Path, states: Sequence[tuple[Observation, list[LegalAction]]],
    variants: Mapping[str, Callable[[Observation], Observation]], *, big_blind: int, starting_stack: int,
    batch: int = 2048,
) -> Comparison:
    """Every state's action distribution once per variant: `variants` maps a name to what
    is done to the state before the model sees it (the legal actions never change)."""
    import torch

    from pokerlab.rl.action_space import legal_action_mask
    from pokerlab.rl.features import encode_observation
    from pokerlab.rl.ppo import build_model_from_checkpoint

    model, _checkpoint = build_model_from_checkpoint(path)
    masks = [legal_action_mask(observation, legal) for observation, legal in states]
    distributions: dict[str, list[list[float]]] = {}
    for name, change in variants.items():
        rows: list[list[float]] = []
        for start in range(0, len(states), batch):
            chunk = range(start, min(start + batch, len(states)))
            features = [
                encode_observation(change(states[i][0]), big_blind=big_blind,
                                   starting_stack=starting_stack, legal_mask=masks[i])
                for i in chunk
            ]
            with torch.no_grad():
                logits, _value = model(torch.tensor(features, dtype=torch.float32),
                                       torch.tensor([masks[i] for i in chunk], dtype=torch.bool))
                rows += torch.softmax(logits, dim=-1).tolist()
        distributions[name] = rows
        print(f"  {name}: fatto", file=sys.stderr, flush=True)
    contexts = [context_of(observation, big_blind) for observation, _legal in states]
    return Comparison(contexts, distributions)


def hud_variant(vector: tuple[float, ...] | None) -> Callable[[Observation], Observation]:
    return lambda observation: with_hud(observation, vector)


def evaluate(path: Path, states: Sequence[tuple[Observation, list[LegalAction]]], profiles: Sequence[Profile],
             population: Population, *, big_blind: int, starting_stack: int, batch: int = 2048) -> Comparison:
    """`evaluate_variants` with one variant per HUD profile."""
    variants = {
        profile.name: hud_variant(None if profile.rates is None else profile_vector(profile.rates, population.per_hand))
        for profile in profiles
    }
    return evaluate_variants(path, states, variants, big_blind=big_blind, starting_stack=starting_stack, batch=batch)


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

    parser = argparse.ArgumentParser(description="come cambia un agente con l'HUD degli avversari")
    parser.add_argument("--model", default="#1", help="etichetta, percorso del checkpoint o #N (rango globale)")
    parser.add_argument("--decisions", type=int, default=20000, help="stati di gioco raccolti in self-play")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--players", type=int, default=None, help="un solo numero di giocatori")
    parser.add_argument("--stack-bb", type=float, default=None, help="stessi stack per tutti")
    parser.add_argument("--session-hands", type=int, default=200,
                        help="mani per tavolo prima di cambiarlo (le statistiche si riempiono su queste)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--root", type=Path, default=Path("checkpoints"), help="la cartella dei checkpoint (per cercare un modello per etichetta)")
    parser.add_argument("--global-dir", type=Path, default=Path("checkpoints/global"))
    parser.add_argument("--min-count", type=int, default=50)
    args = parser.parse_args(argv)

    label, path, rating = resolve_model(args.model, global_dir=args.global_dir, root=args.root)
    if rating is not None:
        label = f"{label} (rating {rating:.0f})"
    snapshot = json.loads((args.global_dir / "registry.json").read_text())
    population = population_from_members(snapshot["members"])
    profiles = profiles_for(population)

    weights = list(DEFAULT_TABLE_WEIGHTS)
    if args.players is not None:
        weights = [1.0 if size == args.players else 0.0 for size in SIZES]
    low, high = (args.stack_bb, args.stack_bb) if args.stack_bb else (DEFAULT_STACK_MIN_BB, DEFAULT_STACK_MAX_BB)
    mix = TableMix(weights=tuple(weights), stack_min_bb=low, stack_max_bb=high,
                   small_blind=DEFAULT_SMALL_BLIND, big_blind=DEFAULT_BIG_BLIND)

    workers = max(1, min(args.jobs, args.decisions))
    rng = random.Random(args.seed)
    shares = [args.decisions // workers + (1 if i < args.decisions % workers else 0) for i in range(workers)]
    jobs = [CollectJob(rng.randrange(2**31), share, str(path), tuple(weights), low, high,
                       mix.small_blind, mix.big_blind, args.session_hands) for share in shares]
    print(f"raccolgo {args.decisions:,} decisioni di {label} contro se stesso...", file=sys.stderr, flush=True)
    states: list[tuple[Observation, list[LegalAction]]] = []
    if workers == 1:
        states = collect_job(jobs[0])
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for part in executor.map(collect_job, jobs):
                states += part
    print(f"valuto {len(profiles)} profili HUD...", file=sys.stderr, flush=True)
    comparison = evaluate(path, states, profiles, population,
                          big_blind=mix.big_blind, starting_stack=mix.starting_stack)
    for line in format_report(comparison, profiles, population, label=label, min_count=args.min_count):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""The search (`rl/search.py`): a decision rebuilt from its observation and played out in sampled
worlds. No torch: the policies here are small functions of the features."""

from __future__ import annotations

import math
import random
import threading
from collections import Counter

import pytest
from support import make_random_legal_bot

from pokerlab.cards.card import Card
from pokerlab.engine.actions import Action, ActionType, LegalAction
from pokerlab.engine.arranged_deal import ArrangingRandom, deal_positions
from pokerlab.engine.config import GameConfig
from pokerlab.engine.state import Street
from pokerlab.engine.table import Table
from pokerlab.players.base import Observation, Player
from pokerlab.players.rl_agent import PolicyDecision, RLAgentPlayer
from pokerlab.rl.action_space import action_index_to_action, legal_action_mask
from pokerlab.rl.search import (
    PROBABILITY_FLOOR,
    BatchGate,
    SearchConfig,
    Searcher,
    SearchPlayer,
    SimModel,
    _reconstruct,
)

BIG_BLIND = 2
STACK = 200
ALL_STREETS = frozenset({Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER})


# ---- policies: small functions of the features -------------------------------------


def _distribution_of(choose):
    def distribution(features, mask, *_style):
        index = choose(features, mask)
        return [1.0 if i == index else 0.0 for i in range(len(mask))]

    return distribution


def deterministic(choose):
    """A policy that always takes `choose(features, mask)`: one-hot probabilities."""

    def policy(features, mask, *_style):
        return PolicyDecision(action_index=choose(features, mask))

    policy.distribution = _distribution_of(choose)
    return policy


def uniform(seed: int):
    rng = random.Random(seed)

    def policy(features, mask, *_style):
        return PolicyDecision(action_index=rng.choice([i for i, ok in enumerate(mask) if ok]))

    def distribution(features, mask, *_style):
        legal = sum(mask)
        return [1.0 / legal if ok else 0.0 for ok in mask]

    policy.distribution = distribution
    return policy


def always_call(features, mask):
    return 1


def pair_raises(features, mask):
    """Raises the minimum with a pair, calls otherwise. Reads its own hole cards off the plane."""
    held = [i for i in range(52) if features[i] == 1.0]
    pair = len(held) == 2 and held[0] % 13 == held[1] % 13
    if pair and mask[2]:
        return 2
    return 1 if mask[1] else 0


def make_player_of(policy):
    return lambda player_id, name: RLAgentPlayer(
        player_id, name, policy_fn=policy, big_blind=BIG_BLIND, starting_stack=STACK
    )


# ---- capturing a real decision ------------------------------------------------------


class Captured(Exception):
    def __init__(self, observation: Observation, legal_actions: list[LegalAction]) -> None:
        super().__init__("captured")
        self.observation = observation
        self.legal_actions = legal_actions


class Scripted(Player):
    """Plays the actions it is given, then (as the hero) captures its decision."""

    def __init__(self, player_id: str, actions: list[Action], *, capture: bool = False) -> None:
        super().__init__(player_id, player_id)
        self._actions = list(actions)
        self._capture = capture

    def act(self, observation, legal_actions):
        if self._actions:
            return self._actions.pop(0)
        if self._capture:
            raise Captured(observation, legal_actions)
        raise AssertionError("the script ran out")


def capture_decision(placements: dict[int, Card], scripts: list[list[Action]], hero: int):
    config = GameConfig(num_players=len(scripts), starting_stack=STACK, small_blind=1, big_blind=BIG_BLIND)
    players = [Scripted(f"p{i}", script, capture=(i == hero)) for i, script in enumerate(scripts)]
    table = Table(config, players, rng=ArrangingRandom(placements, seed=0))
    with pytest.raises(Captured) as captured:
        table.play_hand()
    return captured.value.observation, captured.value.legal_actions


def heads_up_river_with_the_nuts():
    """Seat 0 (button, small blind) holds Jh Th on Ah Kh Qh 2c 3d: a royal flush, which nothing
    beats. Everyone has checked to it, and seat 1 acted first on the river."""
    positions = deal_positions(2)
    cards = {
        **dict(zip(positions["seat0"], [Card.parse("Jh"), Card.parse("Th")], strict=True)),
        **dict(zip(positions["seat1"], [Card.parse("7c"), Card.parse("2d")], strict=True)),
        **dict(zip(positions["flop"], [Card.parse(c) for c in ("Ah", "Kh", "Qh")], strict=True)),
        positions["turn"][0]: Card.parse("2c"),
        positions["river"][0]: Card.parse("3d"),
    }
    call, check = Action(ActionType.CALL), Action(ActionType.CHECK)
    return capture_decision(cards, [[call, check, check], [check, check, check, check]], hero=0)


# ---- the rebuilt decision is the decision that was asked ---------------------------


class Recorder(Player):
    """A random legal bot that remembers what it was asked."""

    def __init__(self, player_id: str, rng: random.Random, sink: list) -> None:
        super().__init__(player_id, player_id)
        self._inner = make_random_legal_bot(player_id, player_id, rng)
        self._sink = sink

    def act(self, observation, legal_actions):
        self._sink.append((observation, legal_actions))
        return self._inner.act(observation, legal_actions)


def decisions_of_random_hands(num_players: int, seed: int, hands: int = 12):
    rng = random.Random(seed)
    sink: list = []
    config = GameConfig(num_players=num_players, starting_stack=STACK, small_blind=1, big_blind=BIG_BLIND)
    table = Table(config, [Recorder(f"p{i}", rng, sink) for i in range(num_players)], rng=rng)
    for _ in range(hands):
        table.stacks = [rng.randint(2 * BIG_BLIND, 3 * STACK) for _ in range(num_players)]
        table.play_hand()
    return sink


@pytest.mark.parametrize("num_players", range(2, 10))
def test_the_rebuilt_hand_asks_the_same_question(num_players):
    """For real decisions at every table size, the replay reaches the hero's turn in a state the
    hero would call identical: the pot, the stacks, the bets, the legal actions, the history."""
    searcher = Searcher(make_player_of(uniform(0)), SearchConfig(), random.Random(1))
    sink = decisions_of_random_hands(num_players, 100 + num_players)
    assert sink
    for observation, legal in sink:
        hand = _reconstruct(observation)
        placements = searcher._sample_world(hand)
        reached = searcher.reach(hand, placements, weigh=False)
        assert reached is not None, f"the replay left the hand on {observation.street}"
        simulated, simulated_legal, _ = reached

        for name in ("street", "pot_size", "current_bet_to_match", "min_raise", "my_stack",
                     "my_current_bet", "hole_cards", "community_cards"):
            assert getattr(simulated, name) == getattr(observation, name), name
        assert simulated_legal == legal
        assert simulated.button_seat == 0 and simulated.my_seat == hand.hero
        assert len(simulated.action_history) == len(observation.action_history)
        for real, rebuilt in zip(observation.action_history, simulated.action_history, strict=True):
            assert hand.sim_of[real.seat] == rebuilt.seat
            assert (real.street, real.action_type, real.amount, real.stack_before, real.stack_after,
                    real.pot_before) == (rebuilt.street, rebuilt.action_type, rebuilt.amount,
                                         rebuilt.stack_before, rebuilt.stack_after, rebuilt.pot_before)
        for seat in observation.seats:
            rebuilt_seat = next(s for s in simulated.seats if s.seat == hand.sim_of[seat.seat])
            assert (seat.stack, seat.current_bet, seat.status) == (
                rebuilt_seat.stack, rebuilt_seat.current_bet, rebuilt_seat.status
            )


def test_the_hole_cards_of_the_opponents_are_drawn_from_the_cards_not_seen():
    observation, _legal = heads_up_river_with_the_nuts()
    searcher = Searcher(make_player_of(uniform(0)), SearchConfig(), random.Random(3))
    hand = _reconstruct(observation)
    seen = {*observation.hole_cards, *observation.community_cards}
    positions = deal_positions(hand.players)
    for _ in range(50):
        placements = searcher._sample_world(hand)
        opponent = [placements[slot] for slot in positions[f"seat{1 - hand.hero}"]]
        assert len(set(opponent)) == 2 and not seen & set(opponent)
        assert len(set(placements.values())) == len(placements), "a card dealt twice"
        assert [placements[slot] for slot in positions[f"seat{hand.hero}"]] == list(observation.hole_cards)


# ---- what it chooses ----------------------------------------------------------------


def test_with_the_nuts_against_a_caller_it_bets_rather_than_checks():
    observation, legal = heads_up_river_with_the_nuts()
    base = make_player_of(deterministic(always_call))("hero", "hero")
    searcher = Searcher(
        make_player_of(deterministic(always_call)),
        SearchConfig(worlds=20, candidates=4, belief="uniform", min_ess=1),
        random.Random(5),
    )
    choice = searcher.choose(observation, legal, base)

    assert choice is not None
    assert choice.policy_top == 1 and choice.changed, "the policy only ever checks"
    check_value = choice.values[1]
    assert choice.action.action_type is ActionType.BET
    assert choice.values[choice.action_index] > check_value
    assert all(value > 0 for value in choice.values.values()), "a royal flush never loses a pot"
    assert choice.worlds == 20 and choice.ess == pytest.approx(20)
    assert searcher.counts.searched == 1 and searcher.counts.changed == 1


def test_every_candidate_is_played_in_the_same_worlds():
    """Paired comparison: the value of a candidate is a mean over the very same worlds, so with
    a caller in the other seat the nuts' value of a bet is the check's plus what the call adds."""
    observation, legal = heads_up_river_with_the_nuts()
    base = make_player_of(deterministic(always_call))("hero", "hero")
    searcher = Searcher(
        make_player_of(deterministic(always_call)),
        SearchConfig(worlds=10, candidates=3, belief="uniform", min_ess=1),
        random.Random(9),
    )
    values = searcher.choose(observation, legal, base).values
    pot_before_the_river = 4  # both limped to 2
    assert values[1] == pytest.approx(pot_before_the_river / 2 / BIG_BLIND)
    bets = [index for index in values if index != 1]
    assert len(bets) == 2
    for index in bets:
        assert values[index] > values[1]


def test_off_its_streets_the_search_leaves_the_decision_to_the_policy():
    observation, legal = heads_up_river_with_the_nuts()
    base = make_player_of(deterministic(always_call))("hero", "hero")
    searcher = Searcher(
        make_player_of(deterministic(always_call)),
        SearchConfig(streets=frozenset({Street.FLOP})),
        random.Random(1),
    )
    assert observation.street is Street.RIVER
    assert searcher.choose(observation, legal, base) is None
    assert searcher.counts.decisions == 0

    player = SearchPlayer("hero", "hero", base, searcher)
    assert player.act(observation, legal).action_type is ActionType.CHECK, "the policy's own call/check"
    assert player.choices == []


def test_a_search_player_plays_whole_hands_and_never_loses_the_replay():
    """The search on every street at every table size, against models that raise and call: the
    replay always gets back to the decision (no world is lost) and the chips are conserved."""
    for num_players in (2, 3, 6, 9):
        rng = random.Random(40 + num_players)
        searcher = Searcher(
            make_player_of(uniform(num_players)),
            SearchConfig(worlds=3, candidates=3, belief="weighted", min_ess=1, streets=ALL_STREETS),
            random.Random(7),
        )
        base = make_player_of(uniform(100 + num_players))("hero", "hero")
        hero = SearchPlayer("hero", "hero", base, searcher)
        players = [hero] + [make_player_of(uniform(i))(f"o{i}", f"o{i}") for i in range(1, num_players)]
        config = GameConfig(num_players=num_players, starting_stack=STACK, small_blind=1, big_blind=BIG_BLIND)
        table = Table(config, players, rng=rng)
        for _ in range(6):
            table.stacks = [rng.randint(2 * BIG_BLIND, 3 * STACK) for _ in range(num_players)]
            before = sum(table.stacks)
            table.play_hand()
            assert sum(table.stacks) == before
        assert searcher.counts.worlds_failed == 0
        assert searcher.counts.decisions > 0


# ---- the belief ---------------------------------------------------------------------


def a_raise_by_a_pair_player():
    """Heads-up preflop: seat 0 (button) raised to the minimum, and it is seat 1's turn."""
    positions = deal_positions(2)
    cards = {
        **dict(zip(positions["seat0"], [Card.parse("7c"), Card.parse("7d")], strict=True)),
        **dict(zip(positions["seat1"], [Card.parse("As"), Card.parse("Kd")], strict=True)),
    }
    return capture_decision(cards, [[Action(ActionType.RAISE, 4)], []], hero=1)


def test_a_world_is_as_likely_as_the_opponents_actions_would_have_been_with_those_cards():
    """The opponent raises only with a pair. A world where it holds a pair keeps its weight;
    one where it holds anything else is a world in which this raise never happens."""
    observation, _legal = a_raise_by_a_pair_player()
    searcher = Searcher(make_player_of(deterministic(pair_raises)), SearchConfig(), random.Random(11))
    hand = _reconstruct(observation)
    positions = deal_positions(2)
    pairs = others = 0
    for _ in range(200):
        placements = searcher._sample_world(hand)
        opponent = [placements[slot] for slot in positions["seat0"]]
        _obs, _legal, log_weight = searcher.reach(hand, placements, weigh=True)
        if opponent[0].rank == opponent[1].rank:
            pairs += 1
            assert log_weight == pytest.approx(0.0)
        else:
            others += 1
            assert log_weight == pytest.approx(math.log(PROBABILITY_FLOOR))
    assert pairs > 0 and others > 0


def test_the_uniform_belief_weighs_every_world_alike_and_does_not_ask_the_models():
    observation, _legal = a_raise_by_a_pair_player()
    searcher = Searcher(
        make_player_of(deterministic(pair_raises)), SearchConfig(belief="uniform"), random.Random(11)
    )
    hand = _reconstruct(observation)
    for _ in range(20):
        _obs, _legal2, log_weight = searcher.reach(hand, searcher._sample_world(hand), weigh=False)
        assert log_weight == 0.0


def test_a_search_with_too_few_effective_worlds_declines():
    """Hands that never raise leave a handful of worlds that count; asked for more than that,
    the search gives the decision back to the policy rather than trust them."""
    observation, legal = a_raise_by_a_pair_player()
    base = make_player_of(deterministic(always_call))("hero", "hero")
    searcher = Searcher(
        make_player_of(deterministic(pair_raises)),
        SearchConfig(worlds=30, candidates=2, belief="weighted", min_ess=1000, streets=ALL_STREETS),
        random.Random(2),
    )
    assert searcher.choose(observation, legal, base) is None
    assert searcher.counts.declined_ess == 1 and searcher.counts.searched == 0


def test_the_opponents_raise_leaves_the_pairs_when_enough_worlds_are_drawn():
    observation, legal = a_raise_by_a_pair_player()
    base = make_player_of(deterministic(always_call))("hero", "hero")
    searcher = Searcher(
        make_player_of(deterministic(pair_raises)),
        SearchConfig(worlds=300, candidates=2, belief="weighted", min_ess=3, streets=ALL_STREETS),
        random.Random(2),
    )
    choice = searcher.choose(observation, legal, base)
    assert choice is not None
    # 78 of the 1,225 hands that can be dealt against As Kd are pairs: about 6% of 300 worlds
    assert 3 <= choice.ess <= 60, "far fewer effective worlds than worlds"


def test_a_search_config_refuses_what_cannot_be_searched():
    with pytest.raises(ValueError, match="belief"):
        SearchConfig(belief="telepathy")
    with pytest.raises(ValueError, match="at least"):
        SearchConfig(candidates=1)


# ---- every world, and the worlds the threshold drops --------------------------------


def test_every_world_of_a_heads_up_river_is_enumerated_once():
    """52 - 2 held - 5 on the board = 45 cards left, so the opponent holds one of C(45, 2) = 990."""
    observation, _legal = heads_up_river_with_the_nuts()
    searcher = Searcher(make_player_of(uniform(0)), SearchConfig(exhaustive=True), random.Random(1))
    hand = _reconstruct(observation)
    positions = deal_positions(2)
    seen = {*observation.hole_cards, *observation.community_cards}

    worlds = list(searcher._all_worlds(hand))
    opponent_hands = {frozenset(world[slot] for slot in positions["seat1"]) for world in worlds}
    assert len(worlds) == 990 == len(opponent_hands)
    assert all(not seen & hands for hands in opponent_hands)
    assert all([world[slot] for slot in positions["seat0"]] == list(observation.hole_cards) for world in worlds)


def test_more_worlds_than_the_cap_are_sampled_instead():
    observation, legal = heads_up_river_with_the_nuts()
    searcher = Searcher(
        make_player_of(deterministic(always_call)),
        SearchConfig(worlds=7, exhaustive=True, max_exhaustive=500, belief="uniform", min_ess=1),
        random.Random(4),
    )
    assert searcher._all_worlds(_reconstruct(observation)) is None  # 990 > 500
    base = make_player_of(deterministic(always_call))("hero", "hero")
    assert searcher.choose(observation, legal, base).worlds == 7, "the sampled worlds of the config"


def test_an_exhaustive_search_plays_every_world_there_is():
    observation, legal = heads_up_river_with_the_nuts()
    base = make_player_of(deterministic(always_call))("hero", "hero")
    searcher = Searcher(
        make_player_of(deterministic(always_call)),
        SearchConfig(exhaustive=True, candidates=2, belief="uniform", min_ess=1),
        random.Random(4),
    )
    choice = searcher.choose(observation, legal, base)
    assert choice.worlds == 990 and searcher.counts.worlds_kept == 990


def test_a_world_that_falls_under_the_threshold_is_dropped_before_it_is_played():
    """The opponent raised and only raises with a pair: a world with anything else holds a hand
    that never takes this line, so it is dropped at the raise and none of its rollouts is played."""
    observation, legal = a_raise_by_a_pair_player()
    base = make_player_of(deterministic(always_call))("hero", "hero")
    searcher = Searcher(
        make_player_of(deterministic(pair_raises)),
        SearchConfig(
            worlds=400, candidates=2, belief="weighted", min_ess=1, world_threshold=0.01, streets=ALL_STREETS
        ),
        random.Random(2),
    )
    choice = searcher.choose(observation, legal, base)

    assert searcher.counts.worlds_pruned > 300, "about 94% of the hands are not a pair"
    assert searcher.counts.worlds_pruned + searcher.counts.worlds_kept == 400
    assert choice.worlds == searcher.counts.worlds_kept
    assert choice.ess == pytest.approx(choice.worlds), "what is left all took the line: equal weights"


def test_reach_reports_a_dropped_world_as_such():
    observation, _legal = a_raise_by_a_pair_player()
    searcher = Searcher(make_player_of(deterministic(pair_raises)), SearchConfig(), random.Random(3))
    hand = _reconstruct(observation)
    positions = deal_positions(2)
    kept = dropped = 0
    for _ in range(100):
        placements = searcher._sample_world(hand)
        reached = searcher.reach(hand, placements, weigh=True, limit=math.log(0.01))
        pair = placements[positions["seat0"][0]].rank == placements[positions["seat0"][1]].rank
        if pair:
            assert reached[0] is not None and reached[2] == pytest.approx(0.0)
            kept += 1
        else:
            assert reached[0] is None and reached[1] is None and reached[2] < math.log(0.01)
            dropped += 1
    assert kept and dropped


def test_a_threshold_is_a_probability_and_needs_the_weighted_belief():
    with pytest.raises(ValueError, match="probability"):
        SearchConfig(world_threshold=1.5)
    with pytest.raises(ValueError, match="weighted belief"):
        SearchConfig(belief="uniform", world_threshold=0.01)


# ---- only the opponents still in the hand are imagined ---------------------------


def three_handed_river_after_a_fold():
    """Seat 0 (button, the hero, Jh Th on Ah Kh Qh 2c 3d: a royal flush) limps; seat 1 folds the
    small blind; seat 2 checks the big blind and checks every street. The river is the hero's."""
    positions = deal_positions(3)
    cards = {
        **dict(zip(positions["seat0"], [Card.parse("Jh"), Card.parse("Th")], strict=True)),
        **dict(zip(positions["seat1"], [Card.parse("7c"), Card.parse("2d")], strict=True)),
        **dict(zip(positions["seat2"], [Card.parse("9s"), Card.parse("8s")], strict=True)),
        **dict(zip(positions["flop"], [Card.parse(c) for c in ("Ah", "Kh", "Qh")], strict=True)),
        positions["turn"][0]: Card.parse("2c"),
        positions["river"][0]: Card.parse("3d"),
    }
    call, check, fold = Action(ActionType.CALL), Action(ActionType.CHECK), Action(ActionType.FOLD)
    return capture_decision(
        cards, [[call, check, check], [fold], [check, check, check, check]], hero=0
    )


def test_a_seat_that_folded_is_not_an_opponent_to_imagine():
    observation, _legal = three_handed_river_after_a_fold()
    searcher = Searcher(make_player_of(uniform(0)), SearchConfig(exhaustive=True), random.Random(1))
    hand = _reconstruct(observation)
    assert hand.live == (hand.sim_of[2],), "only seat 2 is still in"

    worlds = list(searcher._all_worlds(hand))
    assert len(worlds) == 990, "one hand to imagine, not two"
    positions = deal_positions(3)
    for world in worlds[:200]:
        assert len(set(world.values())) == len(world), "a card dealt twice, folded seat included"
        folded_cards = {world[slot] for slot in positions[f"seat{hand.sim_of[1]}"]}
        live_cards = {world[slot] for slot in positions[f"seat{hand.sim_of[2]}"]}
        assert not folded_cards & live_cards


def test_the_actions_of_a_folded_seat_are_not_weighed():
    """Seat 1 folded: whatever its model thinks of that fold, no world is more or less likely
    for it. Only seat 2's checks count, and a model that never raises makes all of them likely."""
    observation, _legal = three_handed_river_after_a_fold()
    searcher = Searcher(make_player_of(deterministic(always_call)), SearchConfig(), random.Random(6))
    hand = _reconstruct(observation)
    for _ in range(20):
        reached = searcher.reach(hand, searcher._sample_world(hand), weigh=True)
        assert reached is not None and reached[0] is not None
        # the fold (bin 0) has probability 0 for a model that always calls: weighed, it would
        # put every world at the floor
        assert reached[2] > math.log(PROBABILITY_FLOOR) * 0.5


def test_the_search_still_plays_out_a_hand_with_a_seat_that_folded():
    observation, legal = three_handed_river_after_a_fold()
    base = make_player_of(deterministic(always_call))("hero", "hero")
    searcher = Searcher(
        make_player_of(deterministic(always_call)),
        SearchConfig(exhaustive=True, candidates=3, belief="uniform", min_ess=1),
        random.Random(8),
    )
    choice = searcher.choose(observation, legal, base)
    assert choice is not None and choice.worlds == 990 and searcher.counts.worlds_failed == 0
    assert choice.action.action_type is ActionType.BET, "the nuts bet against a caller"


# ---- ranges: each opponent weighed on its own, the worlds drawn from them ---------


def a_raise_and_a_call_against_the_hero():
    """Three-handed preflop. Seat 0 (button) raised to the minimum, seat 1 (small blind) called,
    and it is the hero's turn in the big blind with As Kd."""
    positions = deal_positions(3)
    cards = {
        **dict(zip(positions["seat0"], [Card.parse("7c"), Card.parse("7d")], strict=True)),
        **dict(zip(positions["seat1"], [Card.parse("9h"), Card.parse("4s")], strict=True)),
        **dict(zip(positions["seat2"], [Card.parse("As"), Card.parse("Kd")], strict=True)),
    }
    return capture_decision(cards, [[Action(ActionType.RAISE, 4)], [Action(ActionType.CALL)], []], hero=2)


def test_each_opponent_has_a_range_of_its_own():
    """The model raises with a pair and calls with anything else. So the raiser's range is the pairs
    (72: 78 less the six that use the hero's As and Kd) and the caller's is everything that is not
    a pair (1,153 of the 1,225 hands left), each worked out without the other."""
    observation, _legal = a_raise_and_a_call_against_the_hero()
    searcher = Searcher(make_player_of(deterministic(pair_raises)), SearchConfig(ranges=True), random.Random(1))
    hand = _reconstruct(observation)
    raiser, caller = hand.sim_of[0], hand.sim_of[1]
    limit = math.log(0.5)

    raiser_hands, raiser_weights = searcher._range_of(hand, raiser, limit)
    caller_hands, caller_weights = searcher._range_of(hand, caller, limit)

    assert len(raiser_hands) == 72 and all(a.rank == b.rank for a, b in raiser_hands)
    assert len(caller_hands) == 1153 and all(a.rank != b.rank for a, b in caller_hands)
    assert set(raiser_weights) == set(caller_weights) == {1.0}
    assert not {*observation.hole_cards} & {card for pair in raiser_hands for card in pair}


def test_worlds_drawn_from_the_ranges_never_deal_a_card_twice():
    observation, _legal = a_raise_and_a_call_against_the_hero()
    searcher = Searcher(make_player_of(deterministic(pair_raises)), SearchConfig(ranges=True), random.Random(2))
    hand = _reconstruct(observation)
    raiser, caller = hand.sim_of[0], hand.sim_of[1]
    limit = math.log(0.5)
    ranges = {raiser: searcher._range_of(hand, raiser, limit), caller: searcher._range_of(hand, caller, limit)}
    positions = deal_positions(3)

    worlds = list(searcher._joint_worlds(hand, ranges, 200))
    assert len(worlds) == 200
    for world in worlds:
        assert len(set(world.values())) == len(world), "a card dealt twice"
        first, second = (world[slot] for slot in positions[f"seat{raiser}"])
        assert first.rank == second.rank, "the raiser holds a pair"
        first, second = (world[slot] for slot in positions[f"seat{caller}"])
        assert first.rank != second.rank, "the caller does not"


def test_a_search_on_ranges_counts_the_weakest_range_and_plays_the_worlds_asked_for():
    observation, legal = a_raise_and_a_call_against_the_hero()
    base = make_player_of(deterministic(always_call))("hero", "hero")
    searcher = Searcher(
        make_player_of(deterministic(pair_raises)),
        SearchConfig(
            ranges=True, worlds=30, candidates=2, belief="weighted", min_ess=3, world_threshold=0.5,
            streets=ALL_STREETS,
        ),
        random.Random(3),
    )
    choice = searcher.choose(observation, legal, base)
    assert choice is not None
    assert choice.worlds == 30
    assert choice.ess == pytest.approx(72), "the raiser's range is the narrower: 72 equal hands"


def test_ranges_need_the_weighted_belief():
    with pytest.raises(ValueError, match="weighted belief"):
        SearchConfig(ranges=True, belief="uniform")


# ---- spending less: stopping early, dropping losers, folding duplicates, small pots ---------


def test_a_clear_decision_stops_after_the_first_check_and_keeps_the_winner():
    """With the nuts against a caller every bet beats the check by the same amount in every world,
    so the differences have no spread: at the first look every candidate but the biggest bet
    is clearly worse, and nothing more is played."""
    observation, legal = heads_up_river_with_the_nuts()
    base = make_player_of(deterministic(always_call))("hero", "hero")
    searcher = Searcher(
        make_player_of(deterministic(always_call)),
        SearchConfig(worlds=1000, candidates=4, belief="uniform", min_ess=1, stop_z=3.0, min_worlds=10,
                     check_every=10),
        random.Random(5),
    )
    choice = searcher.choose(observation, legal, base)

    assert choice.worlds == 10, "stopped at the first look instead of playing 1,000 worlds"
    assert searcher.counts.rollouts == 4 * 10
    assert searcher.counts.dropped_candidates == 3
    assert choice.action.action_type is ActionType.BET
    assert choice.values[choice.action_index] == max(choice.values.values())


def test_without_stopping_every_world_is_played_for_every_candidate():
    observation, legal = heads_up_river_with_the_nuts()
    base = make_player_of(deterministic(always_call))("hero", "hero")
    searcher = Searcher(
        make_player_of(deterministic(always_call)),
        SearchConfig(worlds=30, candidates=4, belief="uniform", min_ess=1),
        random.Random(5),
    )
    choice = searcher.choose(observation, legal, base)
    assert choice.worlds == 30 and searcher.counts.rollouts == 4 * 30
    assert searcher.counts.dropped_candidates == 0


def test_at_the_river_with_one_opponent_the_worlds_are_its_range_each_hand_once():
    """Nothing is left to come, so a world is the opponent's hand: with duplicates folded the
    search plays the 990 hands it could hold once each, not 1,000 draws with repeats."""
    observation, legal = heads_up_river_with_the_nuts()
    base = make_player_of(deterministic(always_call))("hero", "hero")
    searcher = Searcher(
        make_player_of(deterministic(always_call)),
        SearchConfig(worlds=1000, candidates=2, ranges=True, dedup=True, min_ess=1),
        random.Random(6),
    )
    choice = searcher.choose(observation, legal, base)
    assert choice.worlds == 990 and searcher.counts.rollouts == 2 * 990


def test_duplicate_draws_are_played_once_and_counted_as_many_times():
    observation, _legal = a_raise_and_a_call_against_the_hero()
    searcher = Searcher(
        make_player_of(deterministic(pair_raises)),
        SearchConfig(worlds=300, ranges=True, dedup=True, world_threshold=0.5),
        random.Random(7),
    )
    hand = _reconstruct(observation)
    limit = math.log(0.5)
    ranges = {seat: searcher._range_of(hand, seat, limit) for seat in hand.live}
    worlds = list(searcher._worlds(hand, ranges))
    assert sum(round(math.exp(weight)) for _placements, weight in worlds) == 300, "every draw is counted"
    assert len(worlds) <= 300


def test_with_every_world_and_early_stopping_the_worlds_come_in_a_random_order():
    """Stopping on a prefix of the enumeration would look at the first opponent hands in card order
    only: the worlds are shuffled first, so a prefix is a random sample."""
    observation, _legal = heads_up_river_with_the_nuts()
    searcher = Searcher(
        make_player_of(uniform(0)),
        SearchConfig(exhaustive=True, belief="uniform", stop_z=3.0),
        random.Random(8),
    )
    hand = _reconstruct(observation)
    shuffled = [placements for placements, _weight in searcher._worlds(hand, None)]
    in_order = list(Searcher(make_player_of(uniform(0)), SearchConfig(exhaustive=True), random.Random(8))._all_worlds(hand))
    assert len(shuffled) == len(in_order) == 990
    assert shuffled != in_order
    key = lambda world: tuple(sorted((slot, str(card)) for slot, card in world.items()))
    assert sorted(map(key, shuffled)) == sorted(map(key, in_order))


def test_a_pot_too_small_is_left_to_the_policy():
    observation, legal = heads_up_river_with_the_nuts()  # a pot of 4 chips: 2 big blinds
    base = make_player_of(deterministic(always_call))("hero", "hero")
    searcher = Searcher(
        make_player_of(deterministic(always_call)), SearchConfig(min_pot_bb=5, belief="uniform"), random.Random(1)
    )
    assert searcher.choose(observation, legal, base) is None
    assert searcher.counts.skipped_small_pot == 1 and searcher.counts.decisions == 0

    searcher = Searcher(
        make_player_of(deterministic(always_call)),
        SearchConfig(min_pot_bb=1.5, worlds=5, belief="uniform", min_ess=1),
        random.Random(1),
    )
    assert searcher.choose(observation, legal, base) is not None


def test_spending_settings_that_make_no_sense_are_refused():
    with pytest.raises(ValueError, match="stop_z"):
        SearchConfig(stop_z=-1)
    with pytest.raises(ValueError, match="min_worlds"):
        SearchConfig(min_worlds=0)


# ---- a range carried from one decision of a hand to the next ----------------------


def soft_pairs(features, mask):
    """A policy that leans, not decides: 70% on the min-raise with a pair (when it can), 70% on
    check/call otherwise, the rest spread over what is legal. Every hand keeps some weight."""
    held = [i for i in range(52) if features[i] == 1.0]
    pair = len(held) == 2 and held[0] % 13 == held[1] % 13
    legal = [i for i, ok in enumerate(mask) if ok]
    favourite = 2 if pair and mask[2] else (1 if mask[1] else legal[0])
    rest = 0.3 / max(len(legal) - 1, 1) if len(legal) > 1 else 0.0
    return [(0.7 if i == favourite else rest) if mask[i] else 0.0 for i in range(len(mask))]


def leaning(distribution_of):
    rng = random.Random(0)

    def policy(features, mask, *_style):
        probabilities = distribution_of(features, mask)
        return PolicyDecision(action_index=rng.choices(range(len(mask)), weights=probabilities)[0])

    policy.distribution = lambda features, mask, *_style: distribution_of(features, mask)
    return policy


def flop_and_turn_of_one_hand():
    """Heads-up: the hero (button) limps, the big blind checks; on the flop the big blind bets
    the minimum and the hero calls; on the turn it bets the minimum again. The hero's flop and
    turn decisions, from the same deal."""
    positions = deal_positions(2)
    cards = {
        **dict(zip(positions["seat0"], [Card.parse("Ac"), Card.parse("Qd")], strict=True)),
        **dict(zip(positions["seat1"], [Card.parse("8h"), Card.parse("8s")], strict=True)),
        **dict(zip(positions["flop"], [Card.parse(c) for c in ("Ks", "7d", "2c")], strict=True)),
        positions["turn"][0]: Card.parse("9h"),
        positions["river"][0]: Card.parse("3s"),
    }
    call, check, bet = Action(ActionType.CALL), Action(ActionType.CHECK), Action(ActionType.BET, 2)
    flop = capture_decision(cards, [[call], [check, bet]], hero=0)
    turn = capture_decision(cards, [[call, call], [check, bet, bet]], hero=0)
    return flop, turn


def test_a_range_carried_to_the_next_decision_is_the_range_worked_out_afresh():
    """Carried from the flop to the turn, the range is the one worked out from nothing, and the
    model is only asked about the actions that are new (the turn bet), not again about the old."""
    (flop, _), (turn, _) = flop_and_turn_of_one_hand()
    limit = math.log(0.01)
    asked = Counter()

    def counted(name):
        policy = leaning(soft_pairs)
        inner = policy.distribution

        def distribution(features, mask, *style):
            asked[name] += 1
            return inner(features, mask)

        policy.distribution = distribution
        return policy

    carried = Searcher(make_player_of(counted("carried")), SearchConfig(ranges=True), random.Random(1))
    flop_hand, turn_hand = _reconstruct(flop), _reconstruct(turn)
    (seat,) = turn_hand.live
    carried._range_of(flop_hand, seat, limit)
    asked.clear()
    hands, weights = carried._range_of(turn_hand, seat, limit)

    fresh = Searcher(make_player_of(counted("fresh")), SearchConfig(ranges=True), random.Random(1))
    fresh_hands, fresh_weights = fresh._range_of(turn_hand, seat, limit)

    assert dict(zip(hands, weights, strict=True)) == pytest.approx(dict(zip(fresh_hands, fresh_weights, strict=True)))
    assert asked["carried"] == len(hands) + carried.counts.worlds_pruned, "one question a hand: the turn bet"
    assert asked["fresh"] > 2 * asked["carried"], "afresh, the check and the flop bet are asked again"
    assert not any(Card.parse("9h") in pair for pair in hands), "a hand holding the turn card is out"


def test_a_new_hand_does_not_use_the_last_one_s_ranges():
    (flop, _), _turn = flop_and_turn_of_one_hand()
    _nuts, (_river_obs) = heads_up_river_with_the_nuts()
    searcher = Searcher(make_player_of(leaning(soft_pairs)), SearchConfig(ranges=True), random.Random(1))
    flop_hand = _reconstruct(flop)
    searcher._range_of(flop_hand, flop_hand.live[0], None)
    other = _reconstruct(_nuts)
    before = searcher.counts.range_replays
    searcher._range_of(other, other.live[0], None)
    assert searcher.counts.range_replays - before == 990, "everything weighed again: another hand"


# ---- the first answer played out in full rather than drawn -------------------------------


def uniform_but_always_calls():
    """Says every legal answer is as likely, plays check/call: so a branch is spelled out by the
    probabilities and every branch is then played the same way, and a world has one value."""

    def policy(features, mask, *_style):
        return PolicyDecision(action_index=1 if mask[1] else next(i for i, ok in enumerate(mask) if ok))

    def distribution(features, mask, *_style):
        legal = sum(mask)
        return [1.0 / legal if ok else 0.0 for ok in mask]

    policy.distribution = distribution
    return policy


def test_against_an_answer_that_is_always_the_same_branching_is_the_plain_rollout():
    observation, _legal = heads_up_river_with_the_nuts()
    searcher = Searcher(make_player_of(deterministic(always_call)), SearchConfig(branch=True), random.Random(1))
    hand = _reconstruct(observation)
    placements = searcher._sample_world(hand)
    bet = Action(ActionType.BET, 4)
    assert searcher.branched_rollout(hand, placements, bet) == searcher.rollout(hand, placements, bet)


def test_a_branched_world_is_the_average_of_its_answers_weighed_by_their_probability():
    observation, _legal = heads_up_river_with_the_nuts()
    searcher = Searcher(make_player_of(uniform_but_always_calls()), SearchConfig(branch=True, branch_min=0.0),
                        random.Random(2))
    hand = _reconstruct(observation)
    placements = searcher._sample_world(hand)
    bet = Action(ActionType.BET, 4)

    probe = searcher._probe(hand, placements, bet)
    assert probe.observation.my_seat != hand.hero, "the opponent answers the bet"
    mask = legal_action_mask(probe.observation, probe.legal_actions)
    answers = [action_index_to_action(i, probe.observation, probe.legal_actions) for i, ok in enumerate(mask) if ok]
    assert len(answers) >= 3, "fold, call and at least one raise"
    by_hand = [searcher.rollout(hand, placements, ((hand.hero, bet), (probe.observation.my_seat, a))) for a in answers]

    assert searcher.branched_rollout(hand, placements, bet) == pytest.approx(sum(by_hand) / len(by_hand))
    assert min(by_hand) < max(by_hand), "the answers really differ: a fold does not pay what a call does"


def test_an_action_nobody_answers_is_just_played():
    """The hero checks behind on the river: nobody acts after it, the hand goes to showdown."""
    observation, _legal = heads_up_river_with_the_nuts()
    searcher = Searcher(make_player_of(uniform_but_always_calls()), SearchConfig(branch=True), random.Random(3))
    hand = _reconstruct(observation)
    placements = searcher._sample_world(hand)
    check = Action(ActionType.CHECK)
    assert isinstance(searcher._probe(hand, placements, check), float)
    assert searcher.branched_rollout(hand, placements, check) == searcher.rollout(hand, placements, check)


def test_a_search_with_branching_still_chooses_and_counts_its_probes():
    observation, legal = heads_up_river_with_the_nuts()
    base = make_player_of(deterministic(always_call))("hero", "hero")
    searcher = Searcher(
        make_player_of(uniform_but_always_calls()),
        SearchConfig(worlds=5, candidates=3, belief="uniform", min_ess=1, branch=True),
        random.Random(4),
    )
    choice = searcher.choose(observation, legal, base)
    assert choice is not None and choice.worlds == 5
    assert searcher.counts.probes == 5 * 3
    assert searcher.counts.rollouts > 5 * 3, "a bet is answered in several ways, each played"


# ---- many hands at once: the gate and the threads ---------------------------------------


def test_the_gate_answers_every_thread_its_own_question_in_few_evaluations():
    sizes: list[int] = []

    def evaluate(features, masks):
        sizes.append(len(features))
        return [[sum(row)] for row in features]

    gate = BatchGate(evaluate)
    answers: dict[int, list[float]] = {}
    together = threading.Barrier(8)  # every thread running before anyone asks, as in a search

    def work(thread: int) -> None:
        gate.enter()
        together.wait()
        try:
            answers[thread] = [gate.ask([float(thread), float(step)], [True])[0] for step in range(5)]
        finally:
            gate.leave()

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert all(not thread.is_alive() for thread in threads), "nobody is left waiting"
    assert answers == {i: [float(i + step) for step in range(5)] for i in range(8)}
    assert sizes == [8] * 5, "each round of questions was answered in one evaluation"


def test_the_gate_answers_at_once_when_nobody_runs_in_parallel():
    gate = BatchGate(lambda features, masks: [[1.0] for _ in features])
    assert gate.ask([0.0], [True]) == [1.0] and gate.batches == 1


def test_an_evaluation_that_fails_reaches_whoever_asked():
    def evaluate(features, masks):
        raise RuntimeError("the model broke")

    with pytest.raises(RuntimeError, match="the model broke"):
        BatchGate(evaluate).ask([0.0], [True])


def batched_of(choose) -> SimModel:
    def batch(features, masks):
        return [_distribution_of(choose)(f, m) for f, m in zip(features, masks, strict=True)]

    return SimModel(batch_distribution=batch, feature_version=4, big_blind=BIG_BLIND, starting_stack=STACK)


def test_a_batched_search_in_threads_finds_what_the_plain_one_finds():
    """With models that never draw at random, the threads change nothing: the same worlds (drawn
    before any thread runs), the same values, the same choice."""
    observation, legal = heads_up_river_with_the_nuts()
    base = make_player_of(deterministic(always_call))("hero", "hero")
    config = SearchConfig(worlds=40, candidates=4, belief="uniform", min_ess=1)
    plain = Searcher(make_player_of(deterministic(always_call)), config, random.Random(9)).choose(observation, legal, base)
    batched_searcher = Searcher(None, config, random.Random(9), batched=batched_of(always_call), threads=8)
    batched = batched_searcher.choose(observation, legal, base)
    assert batched.values == pytest.approx(plain.values) and batched.action_index == plain.action_index
    assert batched_searcher._gate.questions > batched_searcher._gate.batches, "the questions went in batches"


def test_a_range_worked_out_in_threads_is_the_same_range():
    observation, _legal = a_raise_and_a_call_against_the_hero()
    hand = _reconstruct(observation)
    limit = math.log(0.5)
    plain = Searcher(make_player_of(deterministic(pair_raises)), SearchConfig(ranges=True), random.Random(1))
    threaded = Searcher(None, SearchConfig(ranges=True), random.Random(1), batched=batched_of(pair_raises), threads=8)
    for seat in hand.live:
        assert threaded._range_of(hand, seat, limit) == plain._range_of(hand, seat, limit)


def test_a_search_needs_a_model_for_its_seats():
    with pytest.raises(ValueError, match="make_player or batched"):
        Searcher(None, SearchConfig())


# ---- stopping at the hero's next decision and asking what it is worth ------------------------


def test_a_rollout_stops_at_the_hero_s_next_decision_and_takes_the_leaf_value():
    """On the flop the hero faces a bet. Calling (or raising, against a model that calls) brings the
    turn and the hero's next decision: that is where the rollout stops and the leaf says 7 bb.
    Folding ends the hand: no leaf, the chips lost (the 1 bb the hero limped)."""
    (flop, legal), _turn = flop_and_turn_of_one_hand()
    asked: list = []

    def leaf_value(features, observation, holes):
        asked.append((observation, holes))
        return 7.0

    base = make_player_of(deterministic(always_call))("hero", "hero")
    searcher = Searcher(
        make_player_of(deterministic(always_call)),
        SearchConfig(worlds=5, candidates=4, belief="uniform", min_ess=1, leaf=True, streets=ALL_STREETS),
        random.Random(1),
        leaf_value=leaf_value,
    )
    choice = searcher.choose(flop, legal, base)

    assert choice.values[1] == 7.0, "the call reaches the turn decision"
    if 0 in choice.values:
        assert choice.values[0] == -1.0, "a fold is played to its end"
    assert searcher.counts.leaves > 0
    hand = _reconstruct(flop)
    for observation, holes in asked:
        assert observation.my_seat == hand.hero and observation.street is Street.TURN
        assert holes[hand.hero] == flop.hole_cards and len(holes) == 2
        assert len({card for pair in holes.values() for card in pair}) == 4


def test_without_a_leaf_value_the_leaf_is_refused():
    with pytest.raises(ValueError, match="leaf_value"):
        Searcher(make_player_of(uniform(0)), SearchConfig(leaf=True))


def test_a_relative_threshold_keeps_the_hands_that_mix_where_an_absolute_one_drops_them():
    """On the turn the big blind has checked, bet the flop and bet the turn. With `soft_pairs` a
    hand that is not a pair took that line with probability ~0.7 x 0.1 x 0.1, under 1%: an absolute
    1% threshold leaves only the pairs, a relative one (1% of the likeliest hand) keeps both, with
    the same weights as no threshold at all."""
    _flop, (turn, legal) = flop_and_turn_of_one_hand()
    base = make_player_of(leaning(soft_pairs))("hero", "hero")

    def search(**settings):
        config = SearchConfig(ranges=True, worlds=20, candidates=2, min_ess=1, streets=ALL_STREETS, **settings)
        searcher = Searcher(make_player_of(leaning(soft_pairs)), config, random.Random(4))
        return searcher.choose(turn, legal, base), searcher.counts

    absolute, absolute_counts = search(world_threshold=0.01)
    relative, relative_counts = search(world_threshold=0.01, relative_threshold=True)
    full, _ = search()

    assert absolute_counts.worlds_pruned > 900, "every hand that is not a pair is dropped"
    assert relative_counts.worlds_pruned == 0
    assert relative.ess == pytest.approx(full.ess)
    assert absolute.ess < relative.ess / 3


def test_a_relative_threshold_needs_ranges():
    with pytest.raises(ValueError, match="needs ranges"):
        SearchConfig(world_threshold=0.01, relative_threshold=True)


def test_each_seat_can_be_believed_to_be_its_own_model():
    """The raiser raises only with a pair. Believed to be a model that always calls, its raise is
    as unlikely with any hand, and its range is everything; named as the model it is, its range
    is the 72 pairs left."""
    observation, legal = a_raise_and_a_call_against_the_hero()
    base = make_player_of(deterministic(always_call))("hero", "hero")
    config = SearchConfig(ranges=True, worlds=20, candidates=2, min_ess=1, streets=ALL_STREETS)

    def search(seat_models):
        searcher = Searcher(make_player_of(deterministic(always_call)), config, random.Random(5))
        return searcher.choose(observation, legal, base, seat_models=seat_models)

    believed = search(None)
    named = search({0: make_player_of(deterministic(pair_raises)), 1: make_player_of(deterministic(pair_raises))})
    assert named.ess == pytest.approx(72)
    assert believed.ess > 1000


def test_the_error_of_a_gap_is_that_of_the_paired_differences():
    differences_of = Searcher._paired_error
    assert differences_of([1.0, 2.0, 3.0], [0.0, 1.0, 2.0], [1.0, 1.0, 1.0]) == 0.0, "a constant gap is sure"
    spread = differences_of([1.0, -1.0, 1.0, -1.0], [0.0, 0.0, 0.0, 0.0], [1.0] * 4)
    assert spread == pytest.approx(math.sqrt(1.0 / 4))
    same = [0.5, 0.7]
    assert differences_of(same, same, [1.0, 1.0]) == 0.0


def test_a_choice_carries_the_error_of_every_candidate_against_the_policy_s():
    observation, legal = heads_up_river_with_the_nuts()
    base = make_player_of(deterministic(always_call))("hero", "hero")
    config = SearchConfig(worlds=40, candidates=3, belief="uniform", min_ess=1)
    choice = Searcher(make_player_of(deterministic(always_call)), config, random.Random(9)).choose(observation, legal, base)
    assert set(choice.errors) == set(choice.values)
    assert choice.errors[choice.policy_top] == 0.0

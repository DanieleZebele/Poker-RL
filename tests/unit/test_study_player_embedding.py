"""`studies/player_embedding`: how a hand becomes what an observer sees of a player, and the
embedder that reads it. The records and the data are torch-free; the model tests need torch."""

from __future__ import annotations

import random

import embed_data
import numpy as np
import pytest
from embed_records import (
    ACTION_CLASSES,
    ACTION_DIM,
    DECISION_DIM,
    MAX_DECISIONS,
    NUM_ACTIONS,
    SHOW_MUCKED,
    SHOW_NONE,
    SHOW_SHOWED,
    STATE_DIM,
    SeatHand,
    action_class,
    seat_hands,
    showdown_view,
    state_features,
    strength_of_cards,
)
from support import make_random_legal_bot

from agent_study import decisions_of
from pokerlab.cards.card import Card
from pokerlab.engine.actions import ActionType
from pokerlab.engine.config import GameConfig
from pokerlab.engine.table import Table


def cards(text: str):
    return tuple(Card.parse(card) for card in text.split())


def played_hands(count: int, seed: int = 3, players: int = 5):
    rng = random.Random(seed)
    bots = [make_random_legal_bot(f"p{seat}", rng=rng) for seat in range(players)]
    table = Table(GameConfig(num_players=players, starting_stack=200, small_blind=1, big_blind=2), bots,
                  rng=random.Random(seed))
    hands = []
    for _ in range(count):
        table.stacks = [rng.randint(20, 200) for _ in range(players)]
        hands.append(table.play_hand().hand_history)
    return hands


def test_the_dimensions_add_up():
    assert DECISION_DIM == STATE_DIM + ACTION_DIM
    assert ACTION_DIM == NUM_ACTIONS + 1 == len(ACTION_CLASSES) + 1


def test_every_decision_is_a_vector_in_range_with_exactly_one_action_class():
    for hand in played_hands(80):
        for decision in decisions_of(hand):
            state = state_features(decision)
            assert len(state) == STATE_DIM and all(0.0 <= value <= 1.0 for value in state)
            assert 0 <= action_class(decision) < NUM_ACTIONS


def test_no_private_card_is_in_what_an_observer_reads():
    """The state of a decision is the same whichever cards the player holds."""
    hand = next(h for h in played_hands(40) if len(decisions_of(h)) >= 3)
    decision = decisions_of(hand)[0]
    changed = type(decision)(**{**decision.__dict__, "hole": cards("2c 7d")})
    assert state_features(changed) == state_features(decision)


def test_every_dealt_seat_has_a_hand_even_with_no_decision():
    for hand in played_hands(60):
        seats = seat_hands(hand, {}, random.Random(0))
        assert set(seats) == set(hand.starting_stacks)
        assert all(len(s.vectors) <= MAX_DECISIONS for s in seats.values())


def test_a_hand_that_ends_with_everyone_folding_shows_nothing():
    for hand in played_hands(80):
        folded = {r.seat for r in hand.actions if r.action_type == ActionType.FOLD}
        if len(folded) == len(hand.starting_stacks) - 1:
            view = showdown_view(hand, {seat: 1.0 for seat in hand.starting_stacks}, random.Random(0))
            assert all(state == SHOW_NONE for state, _won, _shown in view.values())
            assert sum(won for _state, won, _shown in view.values()) == 1
            return
    pytest.fail("no hand ended by folds in the sample")


def test_winners_always_show_and_losers_show_by_their_own_probability():
    showdowns = [h for h in played_hands(200, players=4)
                 if len({r.seat for r in h.actions if r.action_type == ActionType.FOLD}) < len(h.starting_stacks) - 1]
    assert len(showdowns) > 20
    for hand in showdowns:
        for p in (0.0, 1.0):
            view = showdown_view(hand, {seat: p for seat in hand.starting_stacks}, random.Random(1))
            folded = {r.seat for r in hand.actions if r.action_type == ActionType.FOLD}
            for seat, (state, won, shown) in view.items():
                net = hand.final_stacks[seat] - hand.starting_stacks[seat]
                if seat in folded:
                    assert state == SHOW_NONE and shown == 0.0
                elif net >= 0:
                    assert state == SHOW_SHOWED  # the winners (and a chop) show whatever p
                else:
                    assert state == (SHOW_SHOWED if p == 1.0 else SHOW_MUCKED)
                assert won == (1 if net > 0 else 0)


def test_the_strength_of_cards_goes_up_with_what_the_hand_holds():
    pre = strength_of_cards(cards("Ah Ad"), ())
    assert pre > strength_of_cards(cards("7c 2d"), ()) and 0.0 <= pre <= 1.0
    board = cards("Kd 8h 2c")
    assert (strength_of_cards(cards("Kc Qs"), board) > strength_of_cards(cards("As 3c"), board))
    assert strength_of_cards(cards("Kc Kh"), board) > strength_of_cards(cards("Kc Qs"), board)


def pack_some(sessions=3, hands=6, players=4):
    parts = []
    for number in range(sessions):
        rng = random.Random(number)
        played = played_hands(hands, seed=number, players=players)
        sequences = [[] for _ in range(players)]
        for hand in played:
            seats = seat_hands(hand, {s: 0.5 for s in range(players)}, rng)
            for seat in range(players):
                sequences[seat].append(seats[seat])
        parts.append(embed_data.pack(
            sequences, models=[f"m{number}-{seat}" for seat in range(players)],
            p_show=[0.5] * players, styled=[False] * players, session=number, hands=hands,
        ))
    return embed_data.concatenate(parts)


def test_packing_keeps_the_hands_in_order_and_pads_the_decisions():
    arrays = pack_some()
    assert arrays["x"].shape == (12, 6, MAX_DECISIONS, DECISION_DIM)
    assert ((arrays["cls"] >= 0).sum(axis=2) <= MAX_DECISIONS).all()
    empty = arrays["cls"] < 0
    assert not arrays["x"][empty].any()  # padding is zeros
    # A sequence is one seat over the session: its hands are the session's, in order.
    assert (arrays["session"][:4] == 0).all() and (arrays["session"][4:8] == 1).all()


def test_a_shard_round_trips(tmp_path):
    arrays = pack_some()
    embed_data.save_shard(tmp_path / "shard-0000.npz", arrays)
    back = embed_data.load_shards([tmp_path / "shard-0000.npz"])
    for key in embed_data.ARRAY_KEYS:
        assert (back[key] == arrays[key]).all(), key


def test_the_validation_models_are_the_same_in_every_shard_and_never_train():
    labels = [f"model-{i}" for i in range(400)]
    first = embed_data.holdout_mask(labels, 0.2)
    again = embed_data.holdout_mask(list(reversed(labels)), 0.2)[::-1]
    assert (first == again).all()
    assert 0.1 < first.mean() < 0.3
    assert not embed_data.holdout_mask(labels, 0.0).any()


def test_the_showdown_modes_hide_what_a_real_table_would():
    show = np.array([[SHOW_SHOWED, SHOW_SHOWED, SHOW_MUCKED, SHOW_NONE]], dtype=np.int8)
    won = np.array([[1, 0, 0, 0]], dtype=np.int8)
    shown = np.array([[0.8, 0.4, 0.0, 0.0]], dtype=np.float16)
    full = embed_data.apply_mode(show, won, shown, "full")
    assert (full[0] == show).all() and (full[1] == shown).all()
    winners = embed_data.apply_mode(show, won, shown, "winners")
    assert winners[0].tolist() == [[SHOW_SHOWED, SHOW_MUCKED, SHOW_MUCKED, SHOW_NONE]]
    assert winners[1].tolist() == [[pytest.approx(0.8, abs=1e-3), 0.0, 0.0, 0.0]]
    none = embed_data.apply_mode(show, won, shown, "none")
    assert not none[0].any() and not none[1].any()
    assert show[0, 1] == SHOW_SHOWED  # the data itself was not touched


# ---- the model (needs torch) ---------------------------------------------------


torch = pytest.importorskip("torch")


def model_batch(batch=3, hands=8):
    arrays = pack_some(sessions=1, hands=hands, players=batch)
    data = {
        "x": torch.from_numpy(arrays["x"]).float(),
        "mask": torch.from_numpy(arrays["cls"] >= 0),
        "show": torch.from_numpy(arrays["show"].astype(np.int64)),
        "won": torch.from_numpy(arrays["won"].astype(np.int64)),
        "shown": torch.from_numpy(arrays["shown"].astype(np.float32)),
    }
    return data


SEQUENCE_MODELS = ("gru", "mean", "transformer")


@pytest.mark.parametrize("sequence_model", SEQUENCE_MODELS)
def test_the_embedding_has_one_vector_per_hand_and_the_first_is_the_same_for_everyone(sequence_model):
    from embed_model import PlayerEmbedder

    embedder = PlayerEmbedder(hidden=16, embedding=4, sequence_model=sequence_model).eval()
    data = model_batch()
    out = embedder.embeddings(data["x"], data["mask"], data["show"], data["won"], data["shown"])
    assert out.shape == (3, 8, 4)
    # Before any hand a player is unknown: the same vector for all of them.
    assert torch.allclose(out[0, 0], out[1, 0]) and torch.allclose(out[0, 0], out[2, 0])


@pytest.mark.parametrize("sequence_model", SEQUENCE_MODELS)
def test_the_embedding_before_hand_t_does_not_see_hand_t_and_does_see_the_hands_before(sequence_model):
    """The leak that would make the whole study worthless: the hand in progress must not be in
    the embedding that is used to play it -- whichever sequence model reads the hands."""
    from embed_model import PlayerEmbedder

    torch.manual_seed(0)
    embedder = PlayerEmbedder(hidden=16, embedding=4, sequence_model=sequence_model).eval()
    data = model_batch()
    base = embedder.embeddings(data["x"], data["mask"], data["show"], data["won"], data["shown"])

    changed = {key: value.clone() for key, value in data.items()}
    t = 5
    changed["x"][:, t] += 1.0
    changed["show"][:, t] = SHOW_SHOWED
    changed["shown"][:, t] = 0.9
    after = embedder.embeddings(changed["x"], changed["mask"], changed["show"], changed["won"], changed["shown"])

    assert torch.allclose(base[:, : t + 1], after[:, : t + 1], atol=1e-6)  # up to and including hand t
    assert not torch.allclose(base[:, t + 1], after[:, t + 1], atol=1e-4)  # hand t+1 has read it


@pytest.mark.parametrize("sequence_model", SEQUENCE_MODELS)
def test_a_hand_with_no_decision_is_encoded_without_nan(sequence_model):
    from embed_model import PlayerEmbedder

    embedder = PlayerEmbedder(hidden=16, embedding=4, sequence_model=sequence_model)
    data = model_batch()
    data["mask"][:, 2] = False
    out = embedder.embeddings(data["x"], data["mask"], data["show"], data["won"], data["shown"])
    assert torch.isfinite(out).all()


def test_the_predictor_shapes_and_a_training_step_runs():
    from embed_model import PlayerEmbedder, Predictor
    from embed_train import losses

    embedder = PlayerEmbedder(hidden=16, embedding=4)
    predictor = Predictor(embedding=4, hidden=16)
    data = model_batch()
    data["cls"] = torch.from_numpy(np.maximum(pack_some(sessions=1, hands=8, players=3)["cls"], 0).astype(np.int64))
    data["strength"] = torch.rand(data["mask"].shape)
    action, strength, hand_index = losses(embedder, predictor, data, 4)
    assert action.shape == strength.shape == hand_index.shape
    (action.mean() + strength.mean()).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in embedder.parameters())
    # The ablation has no embedder at all and still gives the same shapes.
    without, _, _ = losses(None, predictor, data, 4)
    assert without.shape == action.shape


def test_an_unknown_sequence_model_is_refused():
    from embed_model import PlayerEmbedder

    with pytest.raises(ValueError, match="sequence_model"):
        PlayerEmbedder(sequence_model="lstm")


# ---- the generator: what the models are allowed to see ---------------------------


def _generate_with_recording_policy(opponent_stats: bool, tmp_path):
    """One real session of `embed_generate.play_session` with models that only record what they
    were given: the statistics slots of every seat, as the policy receives them."""
    from embed_generate import Plan, play_session
    from support import fixed_mix

    from pokerlab.rl.features import CARDS_DIM, FIELD_SCALARS_DIM, MAX_SEATS, POT_SCALARS_DIM, SEAT_BASE_FEATURES, SEAT_FEATURES, STREET_DIM
    from pokerlab.rl.policy import PolicyDecision

    start = CARDS_DIM + STREET_DIM + POT_SCALARS_DIM + FIELD_SCALARS_DIM
    stat_slots = [
        start + seat * SEAT_FEATURES + offset
        for seat in range(MAX_SEATS) for offset in range(SEAT_BASE_FEATURES, SEAT_FEATURES)
    ]
    seen: list[float] = []

    def policy(features, mask):
        seen.append(sum(abs(features[i]) for i in stat_slots))
        return PolicyDecision(action_index=next(i for i, ok in enumerate(mask) if ok))

    paths = [str(tmp_path / f"m{i}.pt") for i in range(6)]
    plan = Plan(
        candidates=[(f"m{i}", path) for i, path in enumerate(paths)], mix=fixed_mix(4, stack_bb=100.0),
        session_hands=40, seed=5, style_share=0.0, style_spread=0.5, style_temperature_spread=0.2,
        style_jitter=0.1, style_scales=(1.0,) * 5, opponent_stats=opponent_stats,
    )
    arrays = play_session(plan, 0, {path: policy for path in paths})
    return arrays, seen


def test_by_default_the_models_play_without_seeing_any_opponent_statistics(tmp_path):
    pytest.importorskip("torch")
    arrays, seen = _generate_with_recording_policy(False, tmp_path)
    assert seen and not any(seen)  # every statistic slot of every seat was zero, hand after hand
    assert arrays["x"].shape[:2] == (4, 40)


def test_with_the_statistics_on_they_fill_in_over_the_session(tmp_path):
    pytest.importorskip("torch")
    _arrays, seen = _generate_with_recording_policy(True, tmp_path)
    assert seen[0] == 0.0 and any(seen)

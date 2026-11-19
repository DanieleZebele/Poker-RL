"""`studies/pi_distill`: the teacher is the student plus one projection of the equities (or a copy
of the student plus that projection); with the default settings the update is production PPO; the
teacher's term teaches it to read the equities; a privileged teacher really plays when asked to."""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from pi_distill import (
    ARENA_WRITERS,
    SHARED_WRITERS,
    Distiller,
    Settings,
    both_logits,
    distill_update,
    format_line,
    format_ratings,
    format_summary,
    group_ratings,
    install,
    kl,
    make_teacher,
    parse_line,
    summarise,
)
from support import TINY_EQUITY, fixed_mix, tiny_model

from pokerlab.players.rl_agent import RLAgentPlayer
from pokerlab.rl import train
from pokerlab.rl.action_space import ACTION_DIM
from pokerlab.rl.features import OBS_DIM
from pokerlab.rl.policy import EQUITY_SLOTS
from pokerlab.rl.ppo import PPOConfig, TrainingBatch, ppo_update


def batch(count: int = 64, seed: int = 0, advantages=None, actions=None, extra=None) -> TrainingBatch:
    generator = torch.Generator().manual_seed(seed)
    return TrainingBatch(
        features=torch.rand(count, OBS_DIM, generator=generator),
        masks=torch.ones(count, ACTION_DIM, dtype=torch.bool),
        actions=actions if actions is not None else torch.randint(0, ACTION_DIM, (count,), generator=generator),
        old_log_probs=torch.full((count,), -2.4),
        advantages=advantages if advantages is not None else torch.randn(count, generator=generator),
        returns=torch.randn(count, generator=generator) * 0.3,
        critic_extra=extra if extra is not None else torch.rand(count, EQUITY_SLOTS, generator=generator),
    )


def distiller_for(model, **settings) -> Distiller:
    distiller = Distiller(Settings(**settings))
    distiller.attach(model)
    return distiller


@pytest.mark.parametrize("separate", [False, True])
def test_the_student_is_the_model_and_the_teacher_starts_as_the_student(separate):
    model = tiny_model(hidden=16, num_layers=2, head_layers=1)
    teacher_net = make_teacher(model, separate)
    data = batch()
    student, teacher = both_logits(model, teacher_net, data.features, data.masks, data.critic_extra)
    assert torch.equal(student, model.policy_logits(data.features, data.masks))
    assert torch.equal(student, teacher)
    shared = {id(p) for p in model.parameters()}
    assert not shared & {id(p) for p in teacher_net.parameters()}
    assert len(teacher_net.copied_parameters()) == (len(model.policy_parameters()) if separate else 0)


def test_a_separate_teacher_copies_the_student_as_it_is_when_first_used():
    """poker-train makes the teacher before it loads the checkpoint it resumes from."""
    torch.manual_seed(0)
    model = tiny_model(hidden=16, num_layers=1)
    distiller = distiller_for(model, teacher_weight=1.0, separate=True)
    parent = tiny_model(hidden=16, num_layers=1)
    model.load_state_dict(parent.state_dict())   # what --resume does
    data = batch()
    student, teacher = both_logits(model, distiller.ready(), data.features, data.masks, data.critic_extra)
    assert torch.equal(student, teacher)


def test_a_separate_teacher_reaches_the_student_only_through_the_kl():
    """With its own trunk and head and no KL, the teacher's term leaves the student's update the
    production one."""
    config = PPOConfig(epochs=2, minibatch_size=16)
    results = []
    for update in ("production", "study"):
        torch.manual_seed(0)
        model = tiny_model(hidden=16, num_layers=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        distiller = distiller_for(model, teacher_weight=1.0, teacher_clip=1.0, separate=True)
        torch.manual_seed(5)
        if update == "production":
            ppo_update(model, optimizer, batch(), config)
        else:
            distill_update(distiller, model, optimizer, batch(), config)
        results.append(
            ([p.detach().clone() for p in model.parameters()], distiller.teacher.trunk[0].weight.detach().clone())
        )
    (params_a, teacher_a), (params_b, teacher_b) = results
    assert all(torch.allclose(a, b, atol=1e-7) for a, b in zip(params_a, params_b, strict=True))
    assert not torch.equal(teacher_a, teacher_b)   # the teacher did train


def test_with_the_default_settings_the_update_is_production_ppo(capsys):
    config = PPOConfig(epochs=2, minibatch_size=16)
    results = []
    for update in ("production", "study"):
        torch.manual_seed(0)
        model = tiny_model(hidden=16, num_layers=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        distiller = distiller_for(model)   # before the seed: making the projection draws numbers
        torch.manual_seed(5)
        if update == "production":
            stats = ppo_update(model, optimizer, batch(), config)
        else:
            stats = distill_update(distiller, model, optimizer, batch(), config)
        results.append(([p.detach().clone() for p in model.parameters()], stats))
    (params_a, stats_a), (params_b, stats_b) = results
    assert all(torch.equal(a, b) for a, b in zip(params_a, params_b, strict=True))
    for key, value in stats_a.items():
        assert stats_b[key] == pytest.approx(value), key
    assert "distill" not in capsys.readouterr().out


@pytest.mark.parametrize(("clip", "gap", "separate"), [(None, 0.2, False), (1.0, 0.45, False), (1.0, 0.45, True)])
def test_the_teachers_term_teaches_it_to_read_the_equities_and_the_student_cannot(clip, gap, separate):
    """Action 0 is good exactly when the first equity is high: only the teacher can see it. The
    clip holds the teacher within (1 +- epsilon) of the probability the player gave the action, so
    at the run's own clip it can only move from 0.5 to about 0.6."""
    torch.manual_seed(0)
    model = tiny_model(hidden=16, num_layers=1)
    distiller = distiller_for(
        model, teacher_weight=1.0, student_weight=1.0, teacher_clip=clip, separate=separate
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    count = 256
    generator = torch.Generator().manual_seed(1)
    high = torch.rand(count, generator=generator) > 0.5
    extra = torch.zeros(count, EQUITY_SLOTS)
    extra[:, 0] = high.float()
    actions = torch.randint(0, 2, (count,), generator=generator)
    advantages = torch.where((actions == 0) == high, torch.tensor(1.0), torch.tensor(-1.0))
    data = batch(count, seed=2, advantages=advantages, actions=actions, extra=extra)
    data.features[:] = data.features[0]   # one public state: only the equities differ
    data.masks[:, 2:] = False
    data.old_log_probs[:] = torch.log(torch.tensor(0.5))
    config = PPOConfig(epochs=4, minibatch_size=64, clip_epsilon=0.2, entropy_coefficient=0.0)
    for _ in range(15):
        distill_update(distiller, model, optimizer, data, config)
    with torch.no_grad():
        student, teacher = both_logits(model, distiller.teacher, data.features, data.masks, extra)
        p_teacher = torch.softmax(teacher, -1)[:, 0]
        p_student = torch.softmax(student, -1)[:, 0]
    assert float(p_teacher[high].mean() - p_teacher[~high].mean()) > gap
    assert float(p_student.max() - p_student.min()) < 1e-5


def test_the_kl_pulls_the_student_towards_the_teacher():
    torch.manual_seed(0)
    model = tiny_model(hidden=16, num_layers=1)
    distiller = distiller_for(model, beta_student=5.0, student_weight=0.0)
    with torch.no_grad():
        distiller.teacher.projection.weight.normal_(0, 3.0)   # a teacher that disagrees
    data = batch(128, seed=3)
    optimizer = torch.optim.Adam(model.policy_parameters(), lr=3e-3)

    def distance():
        with torch.no_grad():
            s, t = both_logits(model, distiller.teacher, data.features, data.masks, data.critic_extra)
            return float(kl(torch.log_softmax(s, -1), torch.log_softmax(t, -1)).mean())

    start = distance()
    config = PPOConfig(epochs=4, minibatch_size=32, value_coefficient=0.0, entropy_coefficient=0.0)
    for _ in range(10):
        distill_update(distiller, model, optimizer, data, config)
    assert distance() < 0.5 * start


def test_the_log_line_reads_back():
    before = {"kl_st": 0.0123, "kl_ts": 0.0456, "edge_teacher": 0.0321, "edge_student": -0.0012,
              "kl_st_by_street": {"preflop": 0.001, "river": 0.25}}
    line = format_line(7, before, {"teacher_loss": -0.01})
    assert parse_line(line) == {"iteration": 7, "kl_st": 0.0123, "kl_ts": 0.0456, "kl_st_preflop": 0.001,
                                "kl_st_river": 0.25, "edge_teacher": 0.0321, "edge_student": -0.0012}
    assert parse_line("iter    3  reward ...") is None


@pytest.mark.parametrize(("actor", "separate"), [("teacher", False), ("student", False), ("teacher", True)])
def test_an_iteration_runs_with_the_study_installed_and_the_teacher_plays_when_asked(
    monkeypatch, capsys, actor, separate
):
    monkeypatch.setattr(RLAgentPlayer, "act", RLAgentPlayer.act)
    names = ("make_critic_fns", "make_policy_fn", "ppo_update", *SHARED_WRITERS)
    module = SimpleNamespace(**{name: getattr(train, name) for name in names})
    distiller = install(
        Settings(actor=actor, teacher_weight=1.0, beta_student=0.1, beta_teacher=0.1, separate=separate), module
    )
    for name in names:
        monkeypatch.setattr(train, name, getattr(module, name))

    torch.manual_seed(0)
    model = tiny_model(hidden=16, num_layers=1)
    from pokerlab.rl.equity_net import EquityNet

    net = EquityNet(**TINY_EQUITY, arch="sets")
    net.phi.load_state_dict(model.equity_encoder.phi.state_dict())
    critic = train.make_critic_fns(model, net.requires_grad_(False).eval())
    trainer = train.SelfPlayTrainer(
        fixed_mix(3), train.TrainConfig(hands_per_iteration=20), PPOConfig(epochs=1, minibatch_size=32),
        rng=random.Random(0), model=model, critic=critic,
    )
    if actor == "teacher":
        with torch.no_grad():
            distiller.teacher.projection.weight.normal_(0, 1.0)
    stats = trainer.train_iteration()
    assert distiller.teacher_playing is (actor == "teacher")
    assert stats["decisions"] > 0
    line = next(line for line in capsys.readouterr().out.splitlines() if line.startswith("distill "))
    parsed = parse_line(line)
    if actor == "teacher":
        assert parsed["edge_teacher"] == pytest.approx(0.0, abs=1e-5)   # it played: no edge over itself
    else:
        assert parsed["kl_st"] == 0.0 and parsed["edge_student"] == pytest.approx(0.0, abs=1e-5)
    assert train.publish_model("x.pt") is None
    assert train.run_elo_fill_in(None, None) == (0, 0)
    assert "non eseguito" in capsys.readouterr().out


def test_every_writer_of_the_shared_store_is_replaced():
    from pokerlab.rl import global_arena

    module = SimpleNamespace(**{name: getattr(train, name) for name in (
        "make_critic_fns", "make_policy_fn", "ppo_update", *SHARED_WRITERS)})
    arena = SimpleNamespace(**{name: getattr(global_arena, name) for name in ARENA_WRITERS})
    install(Settings(), module, arena)
    for name in SHARED_WRITERS:
        assert getattr(module, name) is not getattr(train, name), name
    for name in ARENA_WRITERS:
        assert getattr(arena, name) is not getattr(global_arena, name), name


def test_settings_refuse_what_cannot_be_run():
    with pytest.raises(ValueError):
        Settings(actor="both")
    with pytest.raises(ValueError):
        Settings(beta_student=-1.0)
    assert not Settings().uses_teacher and Settings(actor="teacher").uses_teacher




def test_the_projection_trains_at_its_own_scale_of_the_runs_rate():
    model = tiny_model(hidden=16, num_layers=1)
    distiller = distiller_for(model, teacher_weight=1.0, teacher_lr_scale=10.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    distill_update(distiller, model, optimizer, batch(), PPOConfig(epochs=1, minibatch_size=32))
    for group in optimizer.param_groups:
        group["lr"] = 1e-4   # what the trainer does before every update
    distill_update(distiller, model, optimizer, batch(), PPOConfig(epochs=1, minibatch_size=32))
    assert [g["lr"] for g in optimizer.param_groups] == [1e-4, pytest.approx(1e-3)]
    separate = distiller_for(model, teacher_weight=1.0, teacher_lr_scale=10.0, separate=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    distill_update(separate, model, optimizer, batch(), PPOConfig(epochs=1, minibatch_size=32))
    assert [g["lr"] for g in optimizer.param_groups] == [1e-4, pytest.approx(1e-3), 1e-4]


def test_a_log_is_summarised_over_its_last_iterations():
    before = {"kl_st": 0.2, "kl_ts": 0.3, "edge_teacher": 0.01, "edge_student": 0.0,
              "kl_st_by_street": {"preflop": 0.1, "river": 0.5}}
    text = "\n".join([
        "iter    1  reward   -4.00 bb  policy -0.0201  value    0.108  entropy 1.800  kl 0.0292  clip 0.382",
        format_line(1, {**before, "kl_st": 0.0}, {"teacher_loss": 0.0}),
        "iter    2  reward   -2.00 bb  policy -0.0201  value    0.108  entropy 1.600  kl 0.0292  clip 0.382",
        format_line(2, before, {"teacher_loss": 0.0}),
        "iter    3  reward   +1.00 bb  policy -0.0201  value    0.108  entropy 1.400  kl 0.0292  clip 0.382",
        format_line(3, before, {"teacher_loss": 0.0}),
    ])
    summary = summarise(text, last=2)
    assert summary["iterations"] == 3
    assert summary["reward_bb"] == pytest.approx(-0.5)
    assert summary["entropy"] == pytest.approx(1.5)
    assert summary["kl_st"] == pytest.approx(0.2) and summary["kl_st_river"] == pytest.approx(0.5)
    assert "edge_teacher" in format_summary({"a": summary}) and summarise("") == {"iterations": 0}


def test_the_placebo_teacher_cannot_learn_what_the_equities_say(tmp_path):
    """The same task as the real teacher's; with the equities of another decision it cannot tell
    when action 0 is good. And a student written at the asked iterations is a loadable model."""
    torch.manual_seed(0)
    model = tiny_model(hidden=16, num_layers=1)
    distiller = distiller_for(model, teacher_weight=1.0, teacher_clip=1.0, placebo=True,
                              save_at=(2,), save_dir=tmp_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    count = 256
    generator = torch.Generator().manual_seed(1)
    high = torch.rand(count, generator=generator) > 0.5
    extra = torch.zeros(count, EQUITY_SLOTS)
    extra[:, 0] = high.float()
    actions = torch.randint(0, 2, (count,), generator=generator)
    advantages = torch.where((actions == 0) == high, torch.tensor(1.0), torch.tensor(-1.0))
    data = batch(count, seed=2, advantages=advantages, actions=actions, extra=extra)
    data.features[:] = data.features[0]
    data.masks[:, 2:] = False
    data.old_log_probs[:] = torch.log(torch.tensor(0.5))
    config = PPOConfig(epochs=4, minibatch_size=64, entropy_coefficient=0.0)
    for _ in range(15):
        distill_update(distiller, model, optimizer, data, config)
    with torch.no_grad():
        _, teacher = both_logits(model, distiller.teacher, data.features, data.masks, extra)
        p_teacher = torch.softmax(teacher, -1)[:, 0]
    assert abs(float(p_teacher[high].mean() - p_teacher[~high].mean())) < 0.1
    from pokerlab.rl.ppo import build_model_from_checkpoint

    _rebuilt, checkpoint = build_model_from_checkpoint(tmp_path / "student-i0002.pt")
    assert checkpoint["iteration"] == 2 and sorted(p.name for p in tmp_path.iterdir()) == ["student-i0002.pt"]
    with pytest.raises(ValueError):
        Settings(actor="teacher", placebo=True)


def test_ratings_are_grouped_by_arm_and_iteration_and_compared_with_the_baseline():
    def result(arm, seed, iteration, bb):
        return {"checkpoint": f"x/e1/{arm}-s{seed}/students/student-i{iteration:04}.pt", "bb_per_100": bb,
                "rating": 1500 + bb}

    results = [result("base", 1, 40, -20.0), result("base", 2, 40, -10.0),
               result("opsd-b1", 1, 40, -5.0), result("opsd-b1", 2, 40, 5.0),
               {"checkpoint": "elsewhere.pt", "bb_per_100": 0.0, "rating": 1500}]
    groups = group_ratings(results)
    assert sorted(groups) == [("base", 40), ("opsd-b1", 40)]
    assert [r["seed"] for r in groups[("opsd-b1", 40)]] == [1, 2]
    text = format_ratings(groups)
    opsd = next(line for line in text.splitlines() if line.startswith("opsd-b1"))
    assert "+15.0 +- 7.1" in opsd and "-5.0 +5.0" in opsd


# ---- the critic in hindsight ----------------------------------------------------------------


def test_the_control_completes_the_visible_board_at_random_and_keeps_it_through_the_hand():
    from pi_distill import hindsight_board

    runout = (3, 17, 30, 41, 50)
    holes = {0, 1, 13, 14}
    boards = [hindsight_board(runout[:visible], runout, holes) for visible in (0, 3, 4, 5)]
    for visible, board in zip((0, 3, 4, 5), boards, strict=True):
        assert len(set(board)) == 5 and board[:visible] == runout[:visible] and not set(board) & holes
    assert boards[-1] == runout
    assert boards[0] != runout   # a random completion, not the real one (this seed)
    with pytest.raises(ValueError):
        hindsight_board((3, 18, 30), runout, holes)


def test_the_study_switches_the_production_runout_on_or_off(monkeypatch):
    """`now` and `real` are `poker-train`'s own switch; `random` replaces the runout's board."""
    from pokerlab.players import rl_agent
    from pokerlab.rl import global_arena

    # `main` installs the study into the real modules: everything it touches is put back after.
    monkeypatch.setattr(rl_agent, "deal_view", rl_agent.deal_view)
    monkeypatch.setattr(RLAgentPlayer, "act", RLAgentPlayer.act)
    for name in ("make_critic_fns", "make_policy_fn", "ppo_update", *SHARED_WRITERS):
        monkeypatch.setattr(train, name, getattr(train, name))
    for name in ARENA_WRITERS:
        monkeypatch.setattr(global_arena, name, getattr(global_arena, name))
    seen = {}

    def fake_main():
        seen["argv"] = list(__import__("sys").argv)

    monkeypatch.setattr(train, "main", fake_main)
    from pi_distill import main

    for board, flag in (("now", "--no-critic-runout"), ("real", "--critic-runout")):
        main(["--pi-critic-board", board, "--", "--iterations", "1"])
        assert flag in seen["argv"] and "--no-global-elo" in seen["argv"]
    main(["--pi-critic-board", "random", "--", "--iterations", "1"])
    assert "--critic-runout" in seen["argv"]
    assert getattr(rl_agent.deal_view, "__name__", "") == "view"   # the control's board
    with pytest.raises(ValueError):
        Settings(critic_board="real", teacher_weight=1.0)


def test_an_iteration_trains_a_critic_that_reads_both_boards(monkeypatch, tmp_path):
    """`both`: the critic reads 18 numbers (the equities now, then on the whole board), and the
    student it saves is an ordinary model."""
    from pokerlab.players import rl_agent
    from pokerlab.rl.equity_net import EquityNet
    from pokerlab.rl.ppo import build_model_from_checkpoint

    monkeypatch.setattr(rl_agent, "deal_view", rl_agent.deal_view)
    monkeypatch.setattr(RLAgentPlayer, "act", RLAgentPlayer.act)
    names = ("make_critic_fns", "make_policy_fn", "ppo_update", *SHARED_WRITERS)
    module = SimpleNamespace(**{name: getattr(train, name) for name in names})
    install(Settings(critic_board="both", save_at=(1,), save_dir=tmp_path), module)
    for name in names:
        monkeypatch.setattr(train, name, getattr(module, name))
    torch.manual_seed(0)
    model = tiny_model(hidden=16, num_layers=1)
    net = EquityNet(**TINY_EQUITY, arch="sets")
    net.phi.load_state_dict(model.equity_encoder.phi.state_dict())
    critic = train.make_critic_fns(model, net.requires_grad_(False).eval())
    assert model.value_trunk[0].in_features == OBS_DIM - 312 + 2 * EQUITY_SLOTS
    trainer = train.SelfPlayTrainer(
        fixed_mix(3), train.TrainConfig(hands_per_iteration=20, critic_runout=True),
        PPOConfig(epochs=1, minibatch_size=32), rng=random.Random(0), model=model, critic=critic,
    )
    stats = trainer.train_iteration()
    assert stats["decisions"] > 0
    rebuilt, _ = build_model_from_checkpoint(tmp_path / "student-i0001.pt")
    features = torch.rand(4, OBS_DIM)
    masks = torch.ones(4, 11, dtype=torch.bool)
    assert torch.allclose(rebuilt.policy_logits(features, masks), model.policy_logits(features, masks))

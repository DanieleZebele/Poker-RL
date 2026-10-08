# Equity network study

A small network that estimates, for every player still in a hand, the **share of the pot**
they will win (their equity), given their cards, the other players' cards and the board
so far. Purpose: an input for the PPO critic (see "the critic" in CLAUDE.md) that is
cheap enough to compute for every training decision; Monte Carlo with the project's
evaluator costs ~10 ms a decision, a forward pass of this network should cost microseconds.

Decided so far (together, step by step):

- **Output**: one number per player in the hand -- the expected share of the pot, ties
  split -- summing to 1.
- **No dataset**: fresh random deals are drawn continuously. The label of a deal is the
  realised result of one random completion of the board, scored with `pokerlab`'s own
  evaluator. Noisy but unbiased, so the network learns the equity itself.
- **Deals** (`deals.py`): 2 to 9 players, uniform; a weight per street picks how much of
  the board is visible (preflop/flop/turn/river, equal by default).
- One board completion per deal.

`check_deals.py` checks the generator (labels sum to 1, AA against KK, speed).

- **Network** (`model.py`): a set network. A shared encoder turns each player's
  [hole cards | visible board] into a vector, the vectors are summed over the players, an
  aggregation network transforms that sum, and a head scores each player from its own vector
  plus the transformed table; a softmax over the players gives the shares. Order of the
  players does not matter.
- **Loss**: log-likelihood of the realised shares (its minimum is the equity).

## Running

```bash
# first the fixed validation sets, made once and cached in runs/ (see below), then train; --device auto uses a GPU
PYTHONPATH=src:studies/equity_net python studies/equity_net/train.py
PYTHONPATH=src:studies/equity_net python studies/equity_net/train.py --resume   # continue
PYTHONPATH=src:studies/equity_net python studies/equity_net/evaluate.py          # the report
PYTHONPATH=src:studies/equity_net python studies/equity_net/analyze.py           # where it goes wrong
```

The deals are scored by the CPU (the project's evaluator), which is what limits the speed on
a GPU box: roughly 800 deals a second per worker process. `--workers` defaults to the cores
less four. The run is saved at every evaluation.

**The river is not evaluated.** With the board complete the winner is read off the hand evaluator
(exact, ties included), so no network has to estimate it. The validation files still hold the river
deals, and every reader (`validation_set`, `hard_set`) returns the sets without them
(`data.without_river`): 18,000 ordinary deals and a smaller, uneven set of hard cases. Errors are
therefore not comparable with the ones measured before this change, which averaged in a river
error close to zero.

**The validation sets** (`data.py`; never used to train, and made on the first run that needs
them -- `evaluate.py` or `train.py`, with `--val-jobs` processes):

- *Ordinary deals*: 24,000, the same number in each of the 32 cells (2-9 players x 4 streets),
  `runs/validation_exact_<deals>x<runouts>.npz`. **The label is exact from the flop on**: every
  completion of the board is enumerated (at most 990 on the flop, 44 on the turn, one on the
  river), a tie splitting the pot. Only the preflop (1.7 million boards) is sampled, from
  `--val-runouts` (10,000) completions, so only there does a label have noise
  (about 0.005 at p = 0.5) and `evaluate.py`'s "minimo raggiungibile" is a preflop figure.
  Cost: the preflop dominates, about 15 s of CPU per deal; with 24 jobs expect ~30 minutes.
- *Hard cases*: `--hard-deals` (400) of each kind in `data.HARD_TAGS` -- two full houses or
  better, two straights or better, a kicker deciding between two hands of the same category,
  a paired board with two strong hands, a pot contested by two players -- found by drawing
  random deals and keeping the ones that fit, all flop or later and so all exact,
  `runs/hard_<per tag>.npz`. A kind too rare to fill is simply short. `evaluate.py` reports the
  error per kind; `analyze.py --hard` breaks it down like the ordinary deals.

**Options of the network and of the training** (`train.py`; the defaults are the network meant
for the long run: `--arch attn --layers 2 --blocks 2 --ema 0.9999`; the original network is
`--arch sets --blocks 0 --ema 0`): `--arch attn --layers 2` (the players attend to each
other instead of a sum over them), `--blocks 4` (residual blocks in the player encoder),
`--ema 0.9999` (a moving average of the weights, which is what is measured and saved), `--loss brier` (squared
error of the shares instead of the log-likelihood), `--weight-decay`, and the existing
`--batch-size`, `--lr`, `--steps`, `--street-weights`. A checkpoint records its own `config`.
The suits of every deal are always put in a canonical order inside the network (`model.canonical_suits`):
the equity does not depend on which suit is which, and the network need not learn 24 copies of everything.
A checkpoint trained before this (`runs/model.pt`) was not given the canonical order; it still loads.

`analyze.py` breaks the error down by street, by number of players, by preflop hand (all 169),
by made hand after the flop and by true equity, and lists the worst single seats.

Results so far, **on the old validation set** (6,000 uniform deals, every label sampled from 1,000
completions; not comparable with the exact sets above, to be redone):

| run | steps x batch | mean error of a share | uniform guess |
|---|---|---|---|
| `runs/model.pt` (256 wide, 356k parameters) | 60,000 x 4,096 | 0.0178 | 0.1716 |

By street: preflop 0.023, flop 0.024, turn 0.022, river 0.002. The error is flat across table
sizes (0.017 to 0.021); the worst seats are flush and straight-flush boards and direct
comparisons between two strong hands.

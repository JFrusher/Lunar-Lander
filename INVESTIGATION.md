# Investigation: what it took to trust a number

This is the story of a reward-shaping study on `LunarLander-v3`. The
headline result is small and, in the end, boring: a per-step time penalty
of 0.1 makes the PPO controller land **22% faster at no cost to reward or
success rate**.

Getting to a result that boring took four separate methodology fixes.
Three of the four were caught by treating a *suspiciously clean* number as
a bug report against my own analysis. This document is about those, not
about the lander.

**Contents**

1. [The setup](#1-the-setup)
2. [Hypothesis: a flat time penalty](#2-hypothesis-a-flat-time-penalty)
3. [The tell: four identical floats](#3-the-tell-four-identical-floats)
4. [The uncomfortable part](#4-the-uncomfortable-part)
5. [Refusing to build on noise](#5-refusing-to-build-on-noise)
6. [Grading your own homework](#6-grading-your-own-homework)
7. [Systematic follow-through](#7-systematic-follow-through)
8. [Where it landed](#8-where-it-landed)
9. [Techniques, and what I'd do differently](#9-techniques-and-what-id-do-differently)

---

## 1. The setup

Two controllers behind one interface:

- **`HeuristicController`** — a hand-built proportional controller. Eight
  gains, five of them tuned by Monte Carlo search over a Latin Hypercube.
- **`RLAgent`** — PPO via Stable-Baselines3.

Both land reliably. Both land *slowly* — hovering their way down while the
environment's shaping reward, which doesn't charge rent for time, quietly
tolerates it.

| | reward | success | landing steps |
|---|---|---|---|
| Heuristic | 246.8 | 97.3% | 401 |
| PPO | 261.2 | 99.3% | 321 |

The obvious lever: charge for time.

## 2. Hypothesis: a flat time penalty

`TimePenaltyWrapper` subtracts a fixed amount from the reward every
timestep. Six penalty levels, both controllers.

One design decision mattered more than it looked: **the penalty is applied
during training/search only, never during evaluation.** Every number
reported is natural, unpenalized reward. Otherwise higher penalties would
mechanically depress the score and the whole sweep would measure its own
thumb on the scale.

## 3. The tell: four identical floats

The first heuristic sweep came back with this:

```
penalty 0.10  ->  mean_reward 250.53465997611173
penalty 0.20  ->  mean_reward 250.53465997611173
penalty 0.40  ->  mean_reward 250.53465997611173
```

Four of six levels, identical to fifteen decimal places.

Floating-point results do not agree to fifteen places by coincidence. That
isn't a weak signal — it's proof that the thing I thought I was varying
wasn't reaching the computation.

It wasn't. `run_monte_carlo` was seeded once, so every penalty level drew
the **same 200 gain sets**. The penalty never re-searched anything; it only
re-ranked a fixed pool. Past a threshold the same gain set won every time,
because it was the same pool every time.

Looking again at the same run, a second bug: PPO training was unseeded.
The RL half of the sweep showed 0% success at four of six levels and
rewards from −313 to +90. I had been about to interpret the *shape* of that
curve. It was network-init noise.

Both fixed in one commit: vary the search seed per level, fix the PPO seed
across levels. **Opposite fixes, same principle** — the thing under study
varies, everything else holds still.

## 4. The uncomfortable part

The seed fix invalidated a decision that had already shipped.

One commit *earlier*, I had promoted new gains into
`configs/heuristic_gains.json` — taken from the penalty=0.2 level of the
sweep I had just proved was broken.

| | shipped (from the buggy run) | corrected run |
|---|---|---|
| mean_reward | 236.6 | 255.6 |
| success | 86.7% | 100% |

The corrected gains won at all three seeds I spot-checked, so I promoted
them and wrote the comparison into the commit message. The point isn't the
gains. It's that a bug fix has a *blast radius* that reaches backwards
into decisions already made on the old output, and the only way to know is
to go looking.

## 5. Refusing to build on noise

With the seeds fixed, the PPO numbers were still untrustworthy for a
reason nobody had checked: **was 150,000 timesteps enough to train the
thing at all?** The figure was inherited, never validated.

![PPO convergence](docs/media/ppo_convergence.png)

| timesteps | reward | success |
|---|---|---|
| 100,000 | **−713.8** | **0%** |
| 200,000 | −38.4 | 0% |
| 400,000 | 235.4 | 96.7% |
| 1,000,000 | 277.0 | 100% |

PPO isn't merely noisy below 400k — it is **non-functional**. At the sweep's
150k budget it cannot land at all. Every RL number in the study to that
point had been measuring an untrained network's opinion about reward
shaping.

The penalty sweep hadn't been *confounded* by undertraining. It had been
measuring undertraining.

## 6. Grading your own homework

The fourth bug is the one I'm least comfortable about, because nothing
looked wrong.

`run_monte_carlo` ranked 200 gain sets on 30 episodes and reported the
best one's score. That score is **the maximum of 200 noisy estimates** —
so it is biased upward by construction, and the bias grows with the number
of samples. Classic winner's curse. The search was reporting its luck.

Re-scoring each winner on 100 episodes it had never seen:

| samples searched | reported best | held-out | inflation |
|---|---|---|---|
| 125 | 255.2 | 248.9 | +6.3 |
| 250 | 258.5 | 234.6 | +23.9 |
| 500 | 259.7 | 229.1 | **+30.6** |
| 1000 | 262.4 | 245.0 | +17.4 |

This one nearly cost me a wrong conclusion. I had run an 8D gain search
and watched best-found climb steadily with sample count — 255 → 258 → 260
→ 262 — which reads as *"the search hasn't converged, it needs a smarter
optimizer."* That was the trigger condition for building CMA-ES.

On held-out episodes the climb doesn't exist. It was the inflation growing,
not the controller improving. **The smallest search generalized best.**

The fix: rank on the search episodes, then re-score the top-k on held-out
episodes and let *those* pick the winner. It reports an honest number and
picks a genuinely better controller:

| selection | held-out reward | held-out success |
|---|---|---|
| naive argmax | 244.2 | 91% |
| held-out selection | **249.3** | **96%** |
| *what the old code advertised* | *264.3* | — |

Grepping for other callers found the same argmax re-derived inside the
sweep, which would have bypassed the fix entirely. Same bug, two sites.

## 7. Systematic follow-through

Four checks that mostly returned "no". Recorded anyway, because a negative
result you can point at is worth more than an open question.

**Are the search bounds too tight?** The top-20 gain sets clustered toward
the edge of two bounds. I widened both in the direction the data pointed
(*opposite* to the direction I'd predicted in writing) and re-ran. The
newly opened regions attracted 2/20 and 0/20 of the top-20 — against ~0.8
expected by chance. The bounds were fine. "Top-20 mean at 80% of range" is
not the same as "top-20 pinned at 100%".

**Do the 3 untuned attitude gains have anything to give?** No — the
objective surface is flat in them. Three seeds' winners actively disagree
(`ANGULAR_VEL_GAIN` = 1.60 / 0.24 / 1.54) while scoring within 4 reward of
each other.

![gain scatter](docs/media/pid_search_scatter.png)

**Are the PPO hyperparameters worth tuning?** 30 configs at 1M timesteps.
The best, retrained across 3 seeds, ties the defaults on reward (261.3 vs
261.2), loses on success (89.3% vs 99.3%), and is **three times more
seed-variable**. Defaults kept.

That comparison had to be redone once. The first pass concluded "nothing
beat the defaults" by comparing against a default score of 277.0 — which
was itself a lucky single-seed draw. Measured properly the defaults are
261.2 ± 4.8. I had been benchmarking against an inflated number and
reaching the right answer for the wrong reason. Then the *correction*
briefly looked like a reversal, and the reversal was itself a single-seed
artifact. Three times in one phase.

**Does the search need CMA-ES?** No — see §6. The 8D space wasn't
under-covered; it was over-fit. Building a stronger optimizer on an
objective that grades its own homework would have overfit harder, faster.
Skipped, and recorded as skipped.

## 8. Where it landed

![time penalty trade-off](docs/media/time_penalty_tradeoff.png)

Three seeds per point, held-out evaluation, error bars.

**On PPO the penalty works, and there is no trade-off to trade.**

| penalty | reward | success | landing steps |
|---|---|---|---|
| 0.0 | 261.2 ± 4.8 | 99.3% | 321 |
| **0.1** | **272.0 ± 5.3** | **99.0%** | **250** |
| 0.4 | 70.0 ± 17.2 | 13.7% | — |

22% faster, reward slightly *higher*, success unchanged. Every seed lands
faster at 0.1 than that same seed at 0.0. At 0.4 the policy collapses.

**On the heuristic it does nothing at all.** Across the entire penalty
range, landing steps move 401 → 388 against per-point standard deviations
of 8–30. Flat.

That asymmetry has a mechanism, and it's the most interesting thing in the
study. The wrapper shapes what PPO *learns* — it changes the objective the
policy is optimized against. For the heuristic it only re-weights an argmax
over a **fixed pool of 200 already-sampled gain sets**. It applies no
search pressure. It cannot *create* a faster controller, only prefer one
that already happened to be sampled. Same wrapper, same penalty, two
completely different mechanisms — and only one of them is reward shaping.

This also retired a planned phase. An altitude-gated penalty had been
designed to buy speed without paying reward. Nothing is being paid. The
original cost estimate that motivated it — 16% faster for 5.6% reward and
13 points of success — came from the pre-fix, undertrained sweep.

## 9. Techniques, and what I'd do differently

**Reinforcement learning** — PPO (Stable-Baselines3), reward shaping via
environment wrappers, convergence validation before interpretation,
hyperparameter search.

**Classical control** — proportional controller design, Monte Carlo gain
tuning.

**Experimental design** — Latin Hypercube sampling, confound isolation via
explicit seed control, held-out validation against selection bias,
multi-seed error bars, matched-budget comparison (5D vs 8D at equal
samples, not equal wall-time).

**Engineering** — `argparse` CLI + `pyproject.toml` console scripts,
multiprocessing for parallel search, pytest with fast/slow separation,
GitHub Actions CI, timestamped run artifacts.

### What I'd do differently

- **Held-out evaluation from day one.** Three of the four bugs are the same
  species: a number that graded itself. Search episodes and reporting
  episodes should have been separate from the first commit.
- **Error bars before conclusions, not after.** Every single-point estimate
  in this project that looked decisive turned out to be a lucky draw —
  277.0, 269.4, 250.53. The pattern was consistent enough that I should
  have stopped trusting single seeds much earlier than I did.
- **Validate inherited constants.** The 150,000-timestep budget was never
  chosen; it was assumed. It was wrong by a factor of ~7 and silently
  invalidated everything downstream.

### Limitations

- 3 seeds per configuration. Enough to catch the errors above; not enough
  for a confident effect size. 5–10 would be better.
- The penalty sweep's sweet spot (0.1) is identified on a 6-point grid. The
  true optimum is somewhere in 0.05–0.2 and this study doesn't resolve it.
- PPO was still improving at 1M timesteps. The convergence check found the
  floor, not the ceiling.
- One environment, one seed family. Nothing here claims to generalize
  beyond `LunarLander-v3`.

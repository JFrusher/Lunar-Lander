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
9. [Chasing the ceiling](#9-chasing-the-ceiling)
10. [Techniques, and what I'd do differently](#10-techniques-and-what-id-do-differently)

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

*Mean of 3 runs each, evaluated on held-out episodes — by the standards
established later in this document, not the ones I started with.*

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

## 9. Chasing the ceiling

The penalty study answered "does this make it faster." It never answered
"faster compared to what." A 22% improvement is only impressive if you know
how much was available.

### How fast *can* it land?

Model the lander as a point mass under gravity with bounded thrust and solve
the minimum-time descent: free-fall, then brake. The constants come off a
live environment rather than a guess — mass 4.817, gravity −10.0, and a main
engine that nets **+0.158 velocity units per tick over gravity** while
firing.

That last number reframes the problem. The engine is barely stronger than
the fall. Any design that assumes the lander can shed speed at the last
moment is assuming authority it doesn't have.

![speed ceiling](docs/media/speed_ceiling.png)

At the shipped controller's own target descent speed, the floor is ~154
ticks against a measured 250 (PPO) and 401 (heuristic). Real headroom —
roughly 40% still on the table for PPO.

**The model was wrong twice, and said so itself.** The first version
returned a "lower bound" *slower* than the real controllers, which is
impossible by construction. Two causes: braking continuously from the switch
point overshoots into sustained ascent, because the engine outpaces gravity
— so the brake phase has to be self-limiting. And it solved on the *mean*
start state, but the environment's random initial push ranges −3.96 to
+3.82, so the mean is a state no episode ever has. Fixed by solving
per-sample and averaging results, not inputs.

I did not catch either by reading the code. I caught them because the answer
was impossible, which is the same reflex as §3 — the number was wrong in a
way that was *visible*, so it got treated as a bug report.

### Denying the engine on purpose

If hovering is the problem, forbid the fix. Block the main engine for the
first N steps and force an efficient fall. Two gate forms: a fixed step
count, or an altitude threshold.

![lockout sweep](docs/media/lockout_step_sweep.png)

Altitude gating fails outright on both controllers — 0% success for PPO at
every threshold tested. A step gate has a fixed, learnable duration; an
altitude gate's duration depends on how each episode's random push unfolds,
which makes it a moving target.

Step gating works, and **both controllers independently put their optimum at
15 steps**:

| controller | baseline steps | step<15 | change | success |
|---|---|---|---|---|
| Heuristic | 397.4 | 367.3 | −7.6% | 98.0% → **100.0%** |
| RL (PPO) | 321.2 ± 24.1 | 264.6 ± 18.2 | −17.6% (3/3 seeds) | 99.3% → 97.0% |

For the heuristic it's strictly free — better reward, success *and* speed at
once. For PPO it's a real trade: 17.6% speed for 2.3 points of success.

**This inverts §8's asymmetry.** The time penalty worked on PPO and did
nothing for the heuristic. Lockout works on the heuristic and is the weaker,
costlier result on PPO. The mechanism explains why: a penalty shapes what a
policy *learns*, so it needs a learner. A lockout constrains what any
controller *may do*, so it applies to a fixed controller directly — and the
heuristic's untuned reaction to being denied its engine happens to be a
better trajectory than the one it chooses freely.

### The fifth lucky draw

The first pass swept `[20, 50, 100]`. 20 won, 50 broke, and I wrote down
that 20 was the optimum. Both halves of that were wrong.

There is no cliff between 20 and 50 — there's a ramp (100% → 97% → 67% →
48% → 21%), invisible because nothing was sampled in it. And 20 wasn't the
optimum; 15 was. "Best of the one viable value I happened to try" is not the
same claim as "best", and I'd written the second.

Then the finer grid produced a fresh trap. On seed 0, step<10 looked like a
second strong contender at 250 steps. Across three seeds it is *worse than
baseline* (+25.9 steps) with a standard deviation of **105.6** — one seed
produced a 459-step policy. A shorter lockout isn't a gentler version of a
longer one; it's less stable.

That is the fifth single-seed result in this project to evaporate on
repeat, after 277.0, 269.4, 250.53, and step<20's own first pass. The
lesson from §9 of the original write-up was "error bars before conclusions."
I had written that lesson down and then, one phase later, reported a
single-seed grid winner anyway.

### What makes step<15 believable

3/3 seeds is better than step<20's 2/3, but at n=3 with std 39.5 it still
isn't significant alone (sign test p≈0.125). What raises confidence is that
it isn't alone: the heuristic leg has **no training seed** — a deterministic
controller on a fixed episode set — and it landed on 15 independently. Two
measurements with unrelated error sources agreeing on the same value is
worth more than either.

### One more caveat, on a column that lies

`avg_landing_steps` averages over *successful* episodes only. At step<40 the
heuristic lands 48 times in 100; at step<50, 21 times. Those configs look
fast because the episodes that survived are the ones that went well. The
apparent speedup past step<20 is substantially selection effect, not speed —
which is why those points are drawn hollow above and excluded from the
trend line rather than quietly plotted alongside honest ones.

## 10. Techniques, and what I'd do differently

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
  277.0, 269.4, 250.53, then step<20 and step<10 in §9. Five times. I wrote
  this lesson down after the third and reported a single-seed grid winner
  one phase later anyway, which suggests knowing the rule and having the
  reflex are different things. The durable fix is procedural, not
  attitudinal: a sweep should not be *able* to report a winner without a
  repeat.
- **Sample the region where the answer lives.** The lockout grid jumped
  20 → 50 and I read the gap as a cliff. It's a ramp, and the real optimum
  sat in the unsampled interval. A grid whose adjacent points disagree that
  violently is telling you it's too coarse.
- **Validate inherited constants.** The 150,000-timestep budget was never
  chosen; it was assumed. It was wrong by a factor of ~7 and silently
  invalidated everything downstream.

### Limitations

- 3 seeds per configuration. Enough to catch the errors above; not enough
  for a confident effect size. 5–10 would be better. §9's headline
  (step<15, −17.6%) is 3/3 same-sign but not significant on its own.
- The speed ceiling in §9 is a 1-D point mass. It ignores horizontal and
  rotational dynamics entirely, so it is a genuine lower bound but a loose
  one — the gap it shows is an upper estimate of the headroom, not a target.
- The penalty sweep's sweet spot (0.1) is identified on a 6-point grid. The
  true optimum is somewhere in 0.05–0.2 and this study doesn't resolve it.
- PPO was still improving at 1M timesteps. The convergence check found the
  floor, not the ceiling.
- One environment, one seed family. Nothing here claims to generalize
  beyond `LunarLander-v3`.

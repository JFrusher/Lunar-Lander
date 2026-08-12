# Investigation: what it took to trust a number

This is the story of trying to make a `LunarLander-v3` controller land
faster, and of how much of that effort went into establishing that the
numbers were real.

Flight time ended up **52% lower** than where it started. The single largest
contributor was not a reinforcement-learning technique — it was giving the
classical proportional controller different gains at different altitudes,
which took fifteen minutes of sampling and left it matching 1M-timestep PPO
on speed while beating it on reliability.

The interesting part is what it took to believe any of that. Seven separate
small-sample results looked decisive and evaporated on repeat. Five
methodology bugs made it into shipped conclusions before being caught —
most of them by treating a *suspiciously clean* number as a bug report
against my own analysis. Four planned phases were cancelled by arguments
rather than experiments. The metric everything was optimised against turned
out to be about a quarter incompressible. And the "physical floor" I had
been measuring progress against for two arcs was not a floor at all.

This document is about those, not about the lander. If you want the working
method rather than the narrative, it's in [docs/METHOD.md](docs/METHOD.md).

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
10. [Where the speed actually went](#10-where-the-speed-actually-went)
11. [Three phases I didn't run](#11-three-phases-i-didnt-run)
12. [Two results that inverted](#12-two-results-that-inverted)
13. [Building better controllers](#13-building-better-controllers)
14. [Techniques, and what I'd do differently](#14-techniques-and-what-id-do-differently)

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

## 10. Where the speed actually went

Two mechanisms remained: pick the heuristic's gains for *speed* rather than
reward, and anneal the time penalty during training instead of fixing it.
Both worked. Neither mattered as much as what they revealed.

### Ranking the same data by a different question

The time penalty is dead for the heuristic — §8 established the mechanism.
But that finding was about *penalty-based* ranking. Ranking by "fewest steps
subject to a success floor" had never been tried, and 17,851 gain samples
were already sitting in `runs/` from 72 earlier sweeps. No new episodes
required.

It found a genuinely better controller: **374.7 steps at 99.5% success**
against the shipped 396.3 at 100%, on less fuel. Stacked with the 15-step
lockout, **351.6 steps — 11.3% faster on 12.9% less fuel**. The two
mechanisms are close to additive, and the fuel column is what proves the
speed isn't bought by burning harder.

I used **three** disjoint episode sets, not two. Picking the best of twenty
candidates *on the held-out set* makes that set a search set — the same
error one level up. The held-out figure was 9 steps optimistic; the third
set gave the real number.

**The finding that outlived the gains:** a success-rate floor measured on 30
episodes does not survive new episodes.

| floor | search-set basis | held-out success | still met the floor |
|---|---|---|---|
| ≥95% | 29/30 | 90.1% (min 77%) | **3 / 10** |
| ≥99% | 30/30 | 94.4% (min 86%) | **2 / 10** |

Seven of ten "≥95% reliable" gain sets were not. 29/30 has a Wilson interval
reaching down to ~83%, and selecting the extreme of 17,851 such estimates
guarantees the survivors are the lucky ones. The constraint is softer than
it reads — it filters on an estimate, not on reliability.

### The collapse was about *when*, not *how much*

§8 left a real question open: why does a 0.4 penalty destroy the policy when
0.1 is free? One hypothesis — the penalty is shaping the reward before the
policy can land at all, so it never discovers landing.

Annealing the penalty instead of fixing it tests exactly that. Same final
penalty, same budget, same seeds:

| | reward | success | steps |
|---|---|---|---|
| flat 0.4 | 70.0 ± 17.2 | **13.7%** | — |
| curriculum → 0.4, stepped | 266.8 ± 3.9 | **97.7%** | 258.6 |

**13.7% → 97.7%.** The policy operates fine at a 0.4 per-step penalty; it
just cannot *learn to land* while already paying it.

The mechanism is legible against something already measured — PPO can't land
at all below ~400k timesteps. At that point the stepped schedule is charging
0.05, inside the free regime. A linear ramp is charging 0.16, already past
where flat runs showed trouble. Shape beat endpoint, and the two shapes
share endpoints by construction.

Linear didn't merely underperform: per-seed success was **12% / 88% / 77%**.
A coin flip. The plan had called for a single-seed comparison first; running
three from the start cost the same wall time because they fit one batch of
workers. Seed 0 alone would have reported "linear fails outright"; seed 100
alone, "linear works fine."

### And then it stopped mattering

The curriculum applies **four times** the terminal penalty pressure of the
best flat run. It lands *slightly slower*: 258.6 steps against 250.5.

| penalty | landing steps |
|---|---|
| 0.1 | 250.5 |
| 0.2 | 246.3 |
| 0.4 (via curriculum) | 258.6 |

Flat, within noise, across a fourfold change in the thing supposedly driving
it. **Time pressure was no longer the binding constraint.** Something else
was, and the penalty axis had nothing left to give.

### 23% of a landing is already over

Splitting every episode at first leg contact:

| controller | total | flight | settling | settling % |
|---|---|---|---|---|
| Heuristic (shipped) | 378.6 | 312.4 | 66.3 | 18% |
| PPO curriculum | 265.6 | 202.6 | 63.0 | 24% |
| **PPO, lockout-trained** | **241.2** | **185.3** | **55.9** | **23%** |

**An episode doesn't end at touchdown.** It ends when Box2D puts the lander
to sleep, and `b2_timeToSleep` is 0.5 seconds — **25 ticks at 50 FPS, a hard
floor**. Observed settling runs 56–67.

That is the saturation. A time penalty charges for those steps, but no
policy can fly its way out of them; it has already landed. Roughly a
quarter of every episode is incompressible, and the share *grows* as
controllers get faster because the numerator barely moves.

It also corrected an error of my own. The idealized ceiling is a
*flight-time* bound — the model stops at ground contact and knows nothing
about settling. I had been comparing it against episode totals:

| | best measured | ceiling | ratio |
|---|---|---|---|
| what I was comparing | 241.2 total | ~124 | ~1.9× |
| **what's comparable** | **185.3 flight** | **~124** | **1.49×** |

Every "% of ceiling" figure I'd written overstated the available headroom.

## 11. Three phases I didn't run

By this point the remaining plan had four phases in it. Three of them died
to arguments rather than experiments, and writing those arguments down took
less time than any single training run would have.

**Potential-based reward shaping.** I had promoted this one *out* of the
parked list on the strength of its theoretical guarantee: shaping of the
form `F = γΦ(s') − Φ(s)` is policy-invariant, so it cannot cause the
collapse in §10. Writing the plan a second time, the problem surfaced.
Policy invariance means *the shaped MDP has the same optimal policy as the
original*. That is the entire theorem. So it cannot make the lander land
faster — it converges to whatever the unshaped reward already prefers, and
native reward doesn't pay for speed. The property I'd advertised as the
feature is exactly what makes it unable to do the job. The flat time penalty
works *because* it isn't policy-invariant. Same fact, opposite sides.

**Observation feature augmentation.** The proposal was to append five
engineered features — distance to pad, kinetic energy, time-to-impact — to
the 8-dimensional observation. But that observation *is* the full state; the
environment is Markov. Every appended feature is a deterministic function of
inputs the network already receives, so it adds precisely zero information.
Only the inductive bias can change. My earlier objection ("a network can
learn those anyway") was weaker and would have needed an hour of compute to
settle; this one follows from the state representation.

**Hybrid classical/learned controller.** The premise: classical control
handles the efficient fall, the learned policy handles terminal precision,
so split the job. The flight/settling decomposition above kills it — the
learned policy wins flight (185 vs 312) *and* settling (56 vs 66). There is
no segment left for the classical controller to specialise in. Worth noting
that on total landing steps alone the hybrid still looked plausible; only
measuring the two segments separately made the answer visible.

None of these are impressive results. They are, collectively, about four
hours of compute not spent and three plausible-sounding write-ups not
produced. The check that catches them is the same one running through this
whole document: state the mechanism you think you're exploiting, then ask
whether the evidence actually supports it existing.

## 12. Two results that inverted

The last two phases each overturned something I thought was settled.

### Throttle control makes it slower

With reward shaping saturated and the compressible part of the episode still
1.5× above the physical bound, one candidate remained: the discrete action
space. The main engine is full-on or off, and Phase 1 measured it at only
+0.158 velocity units/tick over gravity — no way to ask for less.

The plan called for SAC. SAC changes the action space *and* the algorithm
*and* the on/off-policy family at once, and I'd written that confound into
the plan myself before deciding to run it anyway. Continuous-PPO changes one
thing. Same algorithm, same budget, same seeds, same episodes.

| arm | reward | success | flight | settling | fuel |
|---|---|---|---|---|---|
| **discrete** | **272.0 ± 5.3** | **99.0% ± 0.0** | **196.4 ± 30.8** | **54.1 ± 3.7** | **3518** |
| continuous | 241.4 ± 19.1 | 90.3% ± 9.8 | 209.1 ± 16.1 | 108.3 ± 44.6 | 6018 |

**Flight time is a wash** — the direct test, and it's negative. Actuator
granularity was not the limit.

The margin of defeat is entirely in the settling tail, which *doubles*.
Throttle lets the policy feather — cushion the contact on partial thrust —
so instead of planting the lander and letting Box2D's sleep timer run, it
hovers through touchdown and keeps resetting the 0.5s countdown. The extra
control authority gets spent making the landing gentler, which makes the
episode longer. Fuel is up 71%, consistent with the same story.

An accident worth recording: the discrete arm re-ran a configuration
measured weeks earlier, through a different code path. Reward 272.0 ± 5.3,
success 99.0%, steps 250.5 ± 27.1 — identical to every printed digit, then
and now. Nothing set out to test that, and it's the strongest evidence here
that the seeding discipline actually holds end to end.

### Everything is brittle, and the ranking inverts

Every number in this project came from one physics setting: `gravity=-10.0`,
no wind. Re-evaluating the finished controllers under conditions none of
them were tuned against:

| controller | nominal | grav −11.9 | wind 10 | wind 15 |
|---|---|---|---|---|
| Heuristic (shipped) | 100.0 | 69.0 | 45.0 | 34.0 |
| Heuristic + lockout 15 | 99.0 | 69.0 | 39.0 | 22.0 |
| PPO curriculum | 99.0 | 91.0 | 44.0 | 27.0 |
| **PPO lockout-trained** | **95.0** | 83.0 | **73.0** | **57.0** |

Wind is the failure mode, not gravity. And **the controller with the worst
nominal success is by far the most robust** — 73% at wind 10 where
everything else sits at 39–45%.

The reason is an accident. Training under a forced 15-step engine lockout
means every episode opens with a stretch of uncontrolled drift the policy
has to recover from — structurally the same problem wind poses. A constraint
added purely for *speed* worked as unintentional domain randomisation.

And the same lockout makes the *heuristic* the least robust thing in the
table (39% at wind 10). It can't adapt; the lockout just removes 15 steps of
its authority in exactly the conditions where it needs them most.

Which is §8's asymmetry again, in a new place. The time penalty shaped what
PPO learned and merely re-ranked a fixed pool for the heuristic. The lockout
is *trained under* by one and *imposed on* the other. Both times, a
mechanism a learner can adapt to behaves completely differently from the
same mechanism applied to a fixed controller. Both times it was invisible
until measured.

![frontier](docs/media/frontier.png)

There is no single winner. Which controller is "best" depends on a success
floor I deliberately never fixed — and adding the robustness column changes
the answer again.

## 13. Building better controllers

Everything to this point tuned *knobs* on two existing controllers. With
reward shaping saturated and the action space exonerated, the remaining
options were to fix the yardstick or build something new. Both, as it turned
out.

### The yardstick was wrong, and not in the direction I expected

The idealized bound had been quoted throughout as a floor no controller
could beat. Extending it from a 1-D point mass to include horizontal
translation and rotation was supposed to *close* the gap by admitting that
real landers must also kill lateral drift.

It widened it. Two things I had backwards:

- **Lateral correction is expensive.** The lander starts nearly centred but
  carrying up to 4 u/s sideways, and nulling that on the weak side engines
  costs 118 ticks — comparable to the entire vertical descent.
- **Tilting beats not tilting.** A tilted lander translates using the main
  engine (0.358/tick) instead of the side engines (0.047/tick), and a
  smaller vertical thrust component tracks the touchdown-speed cap more
  tightly where an over-powered engine overshoots below it. The best
  constant tilt lands in **104 ticks against the 1-D model's 124**.

So the 124-tick "floor" was never a floor. It was the best *untilted*
strategy, and every "% of ceiling" figure I had written compared against a
strategy while calling it a bound. The function now returns explicitly
labelled achievable-strategy estimates.

A test caught a related bug while I was at it: the tilt search initially
*scored* infeasible angles instead of skipping them, and preferred them —
past a certain tilt the vertical thrust drops below gravity, and falling out
of the sky reaches the ground quickly.

### Planning: fastest, cheapest, least reliable

Nothing in the repo looked ahead. The bound solver already computes an
optimal descent, so putting it in a receding-horizon loop was a short step:
sample action sequences, score them against the model, execute the first
action, replan.

It took four fixes to work, and the useful part is what each one was:

| fix | success | what was wrong |
|---|---|---|
| initial | 0% | The 25-tick horizon never reaches ground (~190 ticks away). No plan lands, so scored on altitude lost the cheapest plan **dives** — the crash it commits to is beyond the horizon. |
| cost-to-go | 25% | — |
| angular velocity + path tilt | 30%, crashes 15%→0% | I penalised final *angle* but not *angular velocity*. Plans arrived perfectly level while spinning, and kept rotating through contact. |
| brake margin | 35% | The planner always intended to brake *later*; because it replans every step, later never arrived. It touched down at 3–5 u/s against a 1.42 cap. |
| **projected** lateral miss | **72%** | Within 25 ticks `x` barely moves, so penalising *current* `x` gave zero gradient — every plan scored alike and it drifted 4–7 units off-pad. |

Two of those are the same root cause in different axes: **a short horizon
cannot see the consequence it is choosing.** Anything not projected past the
horizon is invisible.

The result is the fastest flight in the project (157.8 steps) on the least
fuel, landing 72% of the time. And its robustness profile is the most
interesting thing here:

| condition | MPC | heuristic | PPO lockout-trained |
|---|---|---|---|
| gravity −8.0 (weak) | **99%** | 100% | 94% |
| nominal | 72% | 100% | 95% |
| gravity −11.9 (strong) | 14% | 69% | 83% |

**MPC is the only controller whose best condition is not nominal.** That
asymmetry is a direct read-out of *which way its model is wrong*: it assumes
it can brake at `engine − gravity` with a margin tuned at nominal. Weaken
gravity and real braking beats the estimate, so its conservatism pays off.
Strengthen it and the margin is not enough. A learned policy has no explicit
model to be wrong, so no preferred direction to fail in.

### The biggest win came from the oldest controller

The proportional controller applied one gain set across the whole descent.
Splitting the five core gains across three altitude bands and searching the
resulting 15-D space took about fifteen minutes.

| | flight | success | reward | fuel |
|---|---|---|---|---|
| flat | 308.3 | 99.5% | 255.6 | 10507 |
| **scheduled** | **188.3** | **99.5%** | **272.9** | **6633** |

**39% faster, better reward, 37% less fuel, identical success.** No trade —
strictly better on every axis measured, and enough to match 1M-timestep PPO
on speed while beating it on reliability.

Why it works is the satisfying part. The search independently produced a
monotone staircase — target descent speed −0.295 up high, −0.137 in the
middle, −0.076 on approach. That is precisely the coast-then-brake profile
the idealized model says is optimal, found from nothing but a
success-constrained speed objective. The flat controller *structurally
cannot express it*: one number has to serve both regimes, so it compromises
at −0.112, too slow to fall efficiently and too fast to approach gently.

It also retroactively explains the engine-lockout result from §9. Blocking
the main engine for 15 steps forced a fast initial descent the flat
controller would never choose — a crude hard-coded stand-in for band 0.
Scheduling is the same idea done properly, which is why it delivers 39%
where the lockout delivered 8%.

The CMA-ES question answered itself along the way. Held-out performance
plateaued between 6000 and 12000 samples, so a directed optimiser was not
justified. Worth noting *why* that was checkable: the **search-set** best
rose monotonically the whole way (265 → 277 → 279 → 280) while held-out
flattened. Judged on the search set I would have built CMA-ES to chase a
rise that wasn't real — the same error §7 records.

### Marking, because scalars keep hiding things

Every finding in this document came from decomposing a number. Total steps
hid the incompressible settling tail. One reward number hid that the penalty
saturates. One gain set hid that the descent has two regimes.

So the last piece is a framework that does that by default: segment each
episode into `DESCENT` / `APPROACH` / `TERMINAL` / `SETTLING`, grade each
part against the idealized plan run from that episode's *own* start state
and segmented identically, and detect named behaviours.

It found two regressions in the controller I had just shipped:

| segment | flat | scheduled |
|---|---|---|
| DESCENT | 0.32 | **0.56** |
| APPROACH | 0.33 | **0.62** |
| TERMINAL | **0.85** | 0.66 |

The flat controller's weakness was never the landing — it scores 0.85 there,
close to the idealized bound. It was the descent, at 0.32. And gain
scheduling bought that descent speed partly by **giving terminal quality
back**, a regression entirely hidden by a 39% improvement in the total. The
scheduled controller also burns 20 wasted main-engine frames per episode
(thrust while already slower than a safe touchdown) where the flat one burns
zero.

Neither is a disaster. Both were invisible to every metric I had used for
three arcs, and both took one run of a report card to surface.

## 14. Techniques, and what I'd do differently

**Reinforcement learning** — PPO (Stable-Baselines3), reward shaping via
environment wrappers, curriculum scheduling, convergence validation before
interpretation, hyperparameter search, discrete vs continuous action-space
comparison at matched algorithm and budget.

**Classical control** — proportional controller design, Monte Carlo gain
tuning, gain scheduling by flight regime, bang-bang minimum-time analysis,
model-predictive control (cross-entropy method, receding horizon) against a
measured planar dynamics model.

**Experimental design** — Latin Hypercube sampling, confound isolation via
explicit seed control, held-out validation against selection bias, a *third*
disjoint set when selecting among many candidates, multi-seed error bars,
matched-budget comparison, conditional gates that can cancel planned work,
off-nominal transfer testing.

**Measurement design** — decomposing aggregate metrics into segments that
can be graded separately; distinguishing compressible from incompressible
quantities; grading against a derived physical reference rather than a
chosen target.

**Engineering** — `argparse` CLI + `pyproject.toml` console scripts,
multiprocessing for parallel search, vectorised rollouts for real-time
planning, pytest with fast/slow separation, GitHub Actions CI, timestamped
run artifacts, reproducible figure generation.

### What I'd do differently

- **Held-out evaluation from day one.** Three of the four bugs are the same
  species: a number that graded itself. Search episodes and reporting
  episodes should have been separate from the first commit.
- **Error bars before conclusions, not after.** Every single-point estimate
  in this project that looked decisive turned out to be a lucky draw —
  277.0, 269.4, 250.53, then step<20 and step<10 in §9, and the linear
  curriculum in §10 would have been a sixth. I wrote this lesson down after
  the third and reported a single-seed grid winner one phase later anyway,
  which suggests knowing the rule and having the reflex are different
  things. The durable fix turned out to be procedural rather than
  attitudinal, and embarrassingly simple: **where the seeds fit one batch of
  parallel workers, the multi-seed run costs the same wall time as the
  single-seed one.** Once I stopped treating the cheap version as the
  default, the problem stopped recurring.
- **Sample the region where the answer lives.** The lockout grid jumped
  20 → 50 and I read the gap as a cliff. It's a ramp, and the real optimum
  sat in the unsampled interval. A grid whose adjacent points disagree that
  violently is telling you it's too coarse.
- **Check that the mechanism you're exploiting exists, before building on
  it.** Three planned phases (§11) died to that question, each in under an
  hour of reading. The habit is worth more than any individual result here:
  a plan that names its mechanism explicitly can be falsified on paper,
  while one that only names its method has to be falsified with compute.
- **Measure what you're actually optimising.** Landing steps were the target
  metric for the entire second half of this project, and roughly a quarter
  of them turned out to be post-touchdown settling that no controller can
  influence. The decomposition took an afternoon and changed which phases
  were worth running at all.
- **Check what your reference actually is.** I compared results against a
  "physical floor" for two arcs. It was the best *untilted* strategy, and a
  tilted one beat it — so it was never a floor, and every percentage quoted
  against it was quoting the wrong thing. A bound should be interrogated as
  hard as a result, and it wasn't, because it felt like infrastructure
  rather than a claim.
- **Build the decomposition first, not last.** The marking framework in §13
  found two regressions in a controller I had already shipped, in one run.
  Every one of the four segments it grades separately was a distinction I
  eventually needed anyway — I just discovered each one the expensive way,
  a phase at a time.
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
  With reward shaping and actuation both ruled out as the binding
  constraint, the residual 1.5× is most likely the model's looseness rather
  than anything left in the controller. That is testable and untested.
- **Everything here is brittle.** §12 measured a 71-point success drop for
  the best controller under wind it was never tuned against. Nothing in this
  project should be read as a claim about robust control; it is a study of
  optimisation against one fixed physics setting.
- Robustness was measured at evaluation only. Whether *training* under
  randomised physics recovers it is unanswered — though the accidental
  result in §12, where a speed constraint produced the largest robustness
  gain measured, suggests it would.
- The penalty sweep's sweet spot (0.1) is identified on a 6-point grid. The
  true optimum is somewhere in 0.05–0.2 and this study doesn't resolve it.
- PPO was still improving at 1M timesteps. The convergence check found the
  floor, not the ceiling.
- One environment, one seed family. Nothing here claims to generalize
  beyond `LunarLander-v3`.

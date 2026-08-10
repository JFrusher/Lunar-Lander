# Time-penalty sweep — design

Date: 2026-08-10

## Purpose

Both controllers currently land successfully but slowly (env reward doesn't
penalize hovering — only fuel use and terminal outcome). This adds a
per-step time penalty during training/tuning, sweeps its magnitude across
both controller types, and produces a plot showing the landing-speed /
reward trade-off so a penalty level can be chosen deliberately.

## Decisions (confirmed with user)

- Penalty coefficients swept: `[0.0, 0.02, 0.05, 0.1, 0.2, 0.4]` (6 levels,
  incl. 0 baseline). Subtracted from reward every timestep during
  training/tuning only — evaluation/reporting always uses natural
  (unpenalized) reward.
- RL: 6 independent PPO training runs, 150,000 timesteps each, one per
  penalty level (the penalty changes what the policy learns, so runs can't
  be shared).
- Heuristic: 6 independent pid-search sweeps (200 samples x 30 episodes
  each), one per penalty level. Every candidate gain-set is still evaluated
  with natural reward; the penalty only changes which candidate is picked
  as "best" (`penalized_score = mean_reward − penalty × avg_steps`).
- `avg_landing_steps` metric = steps averaged **only over successful
  landings** (reward >= 200). Crashes/timeouts tracked separately via
  existing `success_rate_pct` / `crash_rate_pct`, not folded into the
  landing-time number.
- Plot: 2-panel line chart, x = penalty coefficient, one line per
  controller (Heuristic, RL (PPO)); panel 1 = avg landing steps, panel 2 =
  mean natural reward.
- Expected wall-clock cost: ~60-90 minutes (6 heuristic sweeps + 6 PPO
  runs). Run in the background once code is verified — not blocking the
  conversation.

## Components

**`lunar_lander_lab/utils/time_penalty.py`** (new)

- `TIME_PENALTY_COEFS: List[float] = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]`
- `class TimePenaltyWrapper(gym.Wrapper)` — subtracts a fixed
  `penalty_per_step` from `reward` every `step()`. Used only when building
  the training env for RL; heuristic doesn't need it (see below).
- `evaluate_controller_natural(controller, num_episodes, env_name="LunarLander-v3", seed_start=0) -> Dict[str, float]`
  — runs `num_episodes` seeded episodes (seeds `seed_start .. seed_start+num_episodes-1`)
  against the **unwrapped** env, returns `mean_reward`, `success_rate_pct`,
  `crash_rate_pct`, `avg_landing_steps` (NaN if zero successes). Mirrors the
  existing per-purpose eval loops in `evaluation.py`/`pid_search.py` rather
  than introducing a new shared abstraction — consistent with this
  codebase's existing pattern.
- `run_time_penalty_sweep(penalties=TIME_PENALTY_COEFS, rl_timesteps=150_000, pid_samples=200, pid_episodes=30, eval_episodes=30, seed=0, n_jobs=None, output_dir=None) -> pd.DataFrame`
  — orchestrates the full sweep (see Data Flow below), writes results +
  plot, returns the combined results DataFrame.

**`lunar_lander_lab/utils/pid_search.py`** (modified)

- `_evaluate_gain_set`: track `success_steps` (steps for episodes that hit
  the success threshold) alongside the existing `steps_taken`; add
  `avg_steps_success` (NaN if no successes) to the returned per-gain-set
  dict.
- `run_monte_carlo`: add `time_penalty: float = 0.0` param. Best-row
  selection changes from `df.sort_values(["mean_reward", "success_rate_pct"]).iloc[0]`
  to `df.loc[(df["mean_reward"] - time_penalty * df["avg_steps"]).idxmax()]`
  — at `time_penalty=0.0` this selects the same row as before in every
  practical case (continuous-valued mean_reward makes exact ties
  vanishingly unlikely, so dropping the secondary tie-break is
  inconsequential). `sweep_manifest.json` and `best_gains.json` gain a
  `time_penalty` field; `best_gains.json`'s summary gains an
  `avg_steps_success` field, both for traceability when the sweep
  orchestrator reads them back.

**`lunar_lander_lab/controllers/rl_agent.py`** (modified)

- `RLAgent.train()` gets one new optional parameter,
  `env_wrapper: Optional[Callable[[gym.Env], gym.Env]] = None`. When set,
  the training env is built as `env_wrapper(gym.make(env_name))` instead of
  `gym.make(env_name)`. `None` (existing call sites, `cmd_train`) preserves
  current behavior exactly.

**`lunar_lander_lab/cli.py`** (modified)

- New `time-penalty-sweep` subcommand, flags: `--rl-timesteps` (150000),
  `--pid-samples` (200), `--pid-episodes` (30), `--eval-episodes` (30),
  `--seed` (0), `--jobs` (None) — same style/defaults as the existing
  `pid-search` subcommand.

## Data Flow

For each of the 6 penalty coefficients, `run_time_penalty_sweep`:

1. **Heuristic:** calls `run_monte_carlo(n_samples=pid_samples, episodes_per_set=pid_episodes, seed=seed, time_penalty=p, output_dir=<sweep_dir>/heuristic_penalty_<p>, n_jobs=n_jobs)`. Recomputes the same penalized-score argmax over the returned DataFrame to read off that penalty's winning row's `mean_reward`, `success_rate_pct`, `crash_rate_pct`, `avg_steps_success` — no extra evaluation pass needed, the sweep already computed everything with natural reward.
2. **RL:** trains a fresh `RLAgent` with `env_wrapper=lambda env: TimePenaltyWrapper(env, p)`, `total_timesteps=rl_timesteps`, `save_path=f"ppo_penalty_{p}"` (lands in the existing `runs/train/<timestamp>/` convention, unchanged). Then evaluates the trained agent with `evaluate_controller_natural(agent, eval_episodes, seed_start=seed)` — same episode count and seed range as the heuristic sweep's own episodes, for a fair comparison.

Both controllers' rows (`controller`, `penalty`, `mean_reward`,
`success_rate_pct`, `crash_rate_pct`, `avg_landing_steps`, plus a `detail`
column holding the gain values or checkpoint path) are collected into one
`pd.DataFrame`, written to
`runs/time_penalty_sweep/<timestamp>/time_penalty_sweep_results.csv`, and
plotted to `.../time_penalty_tradeoff.png` in the same directory.

## Testing

- `_evaluate_gain_set`'s new `avg_steps_success` and `run_monte_carlo`'s new
  penalized selection are exercised by the existing pid-search smoke test
  pattern (run a tiny sweep, inspect output) — no new test scaffolding.
- `TimePenaltyWrapper` gets a `__main__` self-check in `time_penalty.py`
  using a small stub env (no live Gymnasium env needed — deterministic,
  fast): confirms `step()` returns `reward - penalty_per_step` exactly.
- `evaluate_controller_natural` gets a light real-env smoke check (3
  episodes with `HeuristicController`) in the same self-check block,
  confirming the returned dict has the expected keys and a non-NaN
  `avg_landing_steps` (the heuristic controller reliably lands).

## Out of scope

- No changes to the natural env reward used for `run`/`benchmark`/normal
  `pid-search` — the penalty only applies inside this sweep's own
  training/tuning calls.
- No repeated seeds / multiple trials per penalty level (single run per
  level, as budgeted).
- No new shared "evaluate any controller" abstraction refactored out of
  `evaluation.py` — out of scope for this feature, would be an unrelated
  refactor.

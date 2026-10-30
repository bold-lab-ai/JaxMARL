# JaxMARL — Audit Notes: Bugs, Cleanups, and Feature Ideas

This document is a from-scratch explanation of everything found during a full-repo audit of JaxMARL. Each item explains:
1. **The background concept** you need to understand the issue (assuming no prior context).
2. **What the code is actually doing**, with exact file/line references.
3. **Why that's a problem** (or, for features, why it would help).
4. **How to fix / implement it.**

Items are grouped into three tiers: **Bugs** (confirmed, change behavior), **Cleanups** (things that look like bugs but turned out not to change behavior — still worth fixing), and **Enhancements/Features** (code quality and new capabilities).

---

## PART 1 — BUGS (confirmed, actually change results)

### Bug 1: QMIX-RNN / VDN-RNN compute the wrong "episode boundary mask" when training

**Files:** `baselines/QLearning/qmix_rnn.py:484-491`, `baselines/QLearning/vdn_rnn.py:391` (same pattern), `baselines/QLearning/iql_rnn.py:390` (same pattern, worth double-checking)

#### Background you need

These algorithms are variants of **Q-learning**: they train a neural network `Q(obs, action)` to predict the total future discounted reward of taking `action` in state `obs`. The network here is an **RNN** (recurrent neural network — it keeps a "hidden state" memory that carries information across time steps within an episode), because in partially-observable multi-agent games the agent needs memory of what happened earlier in the episode.

Training uses the standard Q-learning update, called the **TD (temporal-difference) target**:

```
target(t) = reward(t) + gamma * (1 - done(t)) * max_a' Q(obs(t+1), a')
```

The `(1 - done(t))` term is the **bootstrap mask**. `done(t)` is 1 if the episode ended right after the action taken at time `t`. If the episode ended, there is no "next state" to look ahead into, so the formula must *not* add any future value — multiplying by `(1 - done(t)) = 0` correctly zeroes that term out. Getting this mask wrong is one of the most common and most damaging bugs in RL code: it either leaks value across episode boundaries (agent thinks it can keep collecting reward after the episode is already over) or needlessly kills value estimates mid-episode.

Separately: because this is an **RNN**, every time a new episode starts, the hidden state must be **reset to zero** (the agent shouldn't remember events from a different, previous episode). The code therefore also needs to know, at every time step, "did the *previous* step end an episode, so I should reset my hidden state before processing this step?"

#### What the code actually does

Look at how a `Timestep` is built during rollout collection, in `qmix_rnn.py:344-386` (`_step_env`):

```python
def _step_env(carry, _):
    hs, last_obs, last_dones, env_state, rng = carry
    ...
    new_obs, new_env_state, rewards, dones, infos = wrapped_env.batch_step(rng_s, env_state, actions)
    timestep = Timestep(
        obs=last_obs,
        actions=actions,
        rewards=rewards,
        dones=last_dones,      # <-- note: this stores the OLD dones, not the new ones!
        avail_actions=avail_actions,
    )
    return (new_hs, new_obs, dones, new_env_state, rng), (timestep, infos)
```

So at every step, the code stores `last_dones` (the done flag **from one step ago**) into the current timestep's `dones` field — on purpose, because that's exactly what's needed to know "should I reset the RNN hidden state right now." In other words: `timestep[t].dones == done_that_actually_happened_at_time(t-1)`.

This means the array `minibatch.dones["__all__"]` is a **time-shifted-by-one** version of the real episode-termination signal. `dones[t]` tells you about termination at `t-1`, not at `t`.

Now look at the loss function, `qmix_rnn.py:484-491`:

```python
qmix_target = (
    minibatch.rewards["__all__"][:-1]
    + (
        1 - minibatch.dones["__all__"][:-1]
    )  # use next done because last done was saved for rnn re-init
    * config["GAMMA"]
    * qmix_next[1:]  # sum over agents
)
```

`rewards[:-1]` is `reward(0), reward(1), ..., reward(T-2)` — the reward actually received at each of those times. To correctly mask `reward(t)`, you need `done(t)` — whether the episode ended right after that reward. But because `dones` is shifted by one, `done(t)` actually lives at **index `t+1`** of the array (i.e., `dones[1:]`), not at index `t` (`dones[:-1]`).

The comment on that line literally says *"use next done because last done was saved for rnn re-init"* — the author correctly describes what **should** happen, but the code one line below still slices `[:-1]` instead of `[1:]`. The comment and the code disagree with each other. This is the clearest signal that it's an unintentional bug, not a deliberate design choice.

#### Why it matters, concretely

With the current code, `reward(t)` gets masked by `done(t-2)→done(t-1)`'s shifted value instead of the done flag that actually corresponds to `reward(t)`'s own transition. Two concrete failure modes:
- **Right after an episode resets**, the new episode's very first transition gets its bootstrap target wrongly zeroed out (because the previous episode's `done=True` is still sitting at that array index), even though nothing ended there.
- **At the true end of an episode**, the bootstrap mask fails to zero out the target, so the network is taught to add predicted future value *after the episode is already over* — this is "value leakage" and is one of the textbook causes of unstable / wrong Q-learning.

**Impact depends on the environment.** If episodes in a given environment almost always run for exactly the full rollout length (`NUM_STEPS`) and never terminate early (e.g. many fixed-horizon MPE scenarios), then `done=True` only ever shows up at the very last index of the array, which this slicing bug doesn't actually reach — so the practical impact there is close to zero. But in environments where episodes commonly end *before* the rollout horizon (e.g. SMAX combat scenarios, where a battle ends the moment one side is eliminated, often well before `NUM_STEPS=128` elapses), this bug corrupts the bootstrap target at every single one of those early terminations. SMAX is one of the environments QMIX/VDN are benchmarked on in this repo, so this is not a purely theoretical concern.

#### How to fix it

Change the slice from `[:-1]` to `[1:]` so it lines up with `rewards[:-1]` and `qmix_next[1:]`:

```python
qmix_target = (
    minibatch.rewards["__all__"][:-1]
    + (
        1 - minibatch.dones["__all__"][1:]
    )  # dones is shifted by one (stores the done that should trigger an RNN reset
       # at this index), so the done that actually corresponds to rewards[t] lives at dones[t+1]
    * config["GAMMA"]
    * qmix_next[1:]
)
```

Apply the identical fix to `baselines/QLearning/vdn_rnn.py:391` (same pattern, same comment). Then check `baselines/QLearning/iql_rnn.py:390` and `baselines/QLearning/transf_qmix.py` / `shaq.py` (which use a similar `learn_traj.dones["__all__"][:-1]` pattern in a few places) to see if they share the same storage convention — if so, they need the same fix; if they already use `dones[1:]` somewhere (as `transf_qmix.py:877` appears to), compare them side by side to understand which one is actually correct.

**Before merging a fix**, it's worth adding a tiny unit test that doesn't rely on real environments at all: construct a fake `minibatch` by hand with a known reward sequence and a known done at a known position, and assert the computed target is exactly what you'd compute by hand. This is the kind of bug that a "does it crash" test (which is all the current CI has — see Enhancement 14 below) will never catch, because the code runs fine and produces a number — just the wrong one.

---

### Bug 2: `jaxmarl.make("JaxRobotarium_arctic_transport")` crashes with `UnboundLocalError`

**File:** `jaxmarl/registration.py:160`

#### Background you need

`jaxmarl.make(env_id, **env_kwargs)` is the library's main entry point for constructing an environment object from a string name, similar to `gymnasium.make("CartPole-v1")`. Internally it's just a big `if env_id == "...": env = SomeClass(...)` / `elif env_id == "...": env = OtherClass(...)` chain, and at the end it does `return env`. The variable `env` only gets a value if one of the `if`/`elif` branches actually matches.

#### What the code actually does

```python
# jaxmarl/registration.py:146-163
# 10. JaxRobotarium Environments
if env_id.startswith("JaxRobotarium"):
    if not SUBMODULE_ENVIRONMENTS:
        raise ValueError(...)
    if env_id == "JaxRobotarium_navigation":
        env = Navigation(**env_kwargs)
    elif env_id == "JaxRobotarium_discovery":
        env = Discovery(**env_kwargs)
    elif env_id == "JaxRobotarium_material_transport":
        env = MaterialTransport(**env_kwargs)
    elif env_id == "JaxRobotarium_warehouse":
        env = Warehouse(**env_kwargs)
    elif env_id == "JaxRobotarium_ arctic_transport":    # <-- stray space after the underscore
        env = ArcticTransport(**env_kwargs)
    elif env_id == "JaxRobotarium_foraging":
        env = Foraging(**env_kwargs)
    ...
```

But the registry of valid names (`registered_envs`, line 207) spells it **without** the space: `"JaxRobotarium_arctic_transport"`. Walk through what happens when a user calls `jaxmarl.make("JaxRobotarium_arctic_transport")`:

1. `env_id.startswith("JaxRobotarium")` → `True`, so we enter this block.
2. None of the string-equality checks match, because the code is comparing against `"JaxRobotarium_ arctic_transport"` (with a space) while `env_id` is `"JaxRobotarium_arctic_transport"` (no space). Python string equality is exact, character-for-character — a single extra space makes two strings completely unequal.
3. We fall through the whole `if`/`elif` chain without ever assigning `env`.
4. At the end of the function, `return env` tries to read a local variable that was never assigned → Python raises `UnboundLocalError: local variable 'env' referenced before assignment`.

This is a 100%-reproducible crash for this exact environment name. Its real-world blast radius is small because `JaxRobotarium*` environments are an optional external submodule (`SUBMODULE_ENVIRONMENTS`, gated by `jaxrobotarium` being installed) that most users won't have installed — but for anyone who *does* use it, this environment is simply unreachable through `jaxmarl.make()`.

#### How to fix it

Remove the stray space:

```python
elif env_id == "JaxRobotarium_arctic_transport":
    env = ArcticTransport(**env_kwargs)
```

That's the entire fix — one character. (See Enhancement 6 below for a structural suggestion that would make this class of typo impossible to write in the first place.)

---

## PART 2 — THINGS THAT LOOK LIKE BUGS BUT AREN'T (worth knowing, worth cleaning up)

These were flagged during the first pass of this audit as "silent bugs," but reading more of the surrounding code showed they don't actually produce wrong results for any caller. Recorded here so nobody re-discovers them and panics, and because the dead code itself is still worth tidying.

### Cleanup 1: JaxNav's `do_sep_reward` flag looks broken but is actually just an unfinished, explicitly-guarded feature

**File:** `jaxmarl/environments/jaxnav/jaxnav_env.py:349-356` and `:393-394`

#### What's there

```python
# lines 349-356
if self.do_sep_reward:
    rew_batch = (
        self.rew_lambda * rew_individual + (1 - self.rew_lambda) * shared_rew
    )
else:
    rew_batch = (
        self.rew_lambda * rew_individual + (1 - self.rew_lambda) * shared_rew
    )
```

Both branches of this `if`/`else` compute the exact same expression — so the `do_sep_reward` flag has no effect on `rew_batch` at all. On its own, that looks like a bug: a documented constructor flag that silently does nothing.

**But** — further down in the same function, before this `rew_batch` value is ever returned to a caller:

```python
# lines 393-394
if self.do_sep_reward:
    raise NotImplementedError("Separate reward not implemented")
```

So the actual behavior is: if you set `do_sep_reward=True`, you get a loud, immediate `NotImplementedError` — the "separate reward" code path was started (the scaffolding exists: the `if`/`else`, a constructor parameter, a commented-out assertion at line 209-210 referencing it) but was never finished, and the author correctly fenced it off so nobody can silently rely on it. Since the default is `do_sep_reward=False`, and the `True` path always raises before returning anything, **no caller can ever observe the "wrong" identical-branches result** — it's unreachable, not silently incorrect.

#### Why it's still worth fixing

It's dead, confusing code: a reader has to trace all the way to line 393 to realize the `if`/`else` at 349 is pointless. Two reasonable options:
- **Simplest:** collapse the two identical branches into one (delete the `if self.do_sep_reward:`/`else:` wrapper, just keep the single formula), since right now it implies a distinction that doesn't exist.
- **Better, if someone wants to actually finish the feature:** implement what "separate reward" is supposed to mean (per the commented-out assertion at line 209-210, it's related to `share_only_sparse` needing `do_sep_reward=True` to work correctly) and make the two branches genuinely different, then remove the `NotImplementedError` guard.

Either way, this is a code-quality fix, not a "numbers are wrong" bug.

### Cleanup 2: Overcooked's visualizer has known, author-acknowledged multi-agent rendering bugs (cosmetic only)

**File:** `jaxmarl/viz/overcooked_visualizer.py:95`, `:323`

#### What's there

```python
# line 95
if highlight:
    # TODO: Fix this for multiple agents
    f_vec = state.agent_dir
    ...
```

```python
# line 323
# TODO: Fix this for multiagents. Currently the orientation of other agents is wrong
if len(agent_dir_idx) == 1:
    # Hacky way of making agent views orientations consistent with global view
    obj[-1] = agent_dir_idx[0]
```

#### What this means

These TODOs are **written by the original authors**, not something hidden that this audit uncovered. They affect two specific pieces of the Overcooked debug/visualization renderer when there are more than two agents on screen: (1) the "highlighted" field-of-view overlay used for debugging, and (2) the on-screen rotation/orientation icon drawn for *other* agents in an individual agent's partial-observation view. Both are purely cosmetic — they affect what a human sees when rendering a trajectory for debugging, not the environment's `step`/`reset` logic, rewards, observations the policy actually receives, or training results in any baseline script.

#### How to fix it

Not attempted here since it requires understanding the visualizer's grid/rotation math in detail, but it's a reasonable "good first issue" for a contributor: the fix would involve computing the highlight bounding box and orientation icon per-agent (indexed by agent ID) instead of assuming a single global `state.agent_dir`.

### Cleanup 3: README doesn't mention that `mabrax` is deprecated and known to crash

**Files:** `README.md:52` (no warning), vs. `CHANGELOG.md:27-30` and `tests/brax/test_brax_rand_acts.py:16-18` (already documented)

#### What's there

The project's own `CHANGELOG.md` already says, in plain language:

> **MABrax**: Deprecated due to Brax itself being deprecated; this environment needs migrating to MJX. ... brax install moved to optional extras ... `jaxmarl.environments.mabrax` will raise an ImportError if brax is not installed.

And the only test for it is explicitly marked:
```python
@pytest.mark.skip(reason="brax legacy environments are unmaintained and crash on current jaxlib")
```

So the maintainers already know and have already documented this. The only actual gap is that `README.md`'s environment table (line 52) lists "🦾 Multi-Agent Brax" like every other fully-supported environment, with no indication that a new user installing it is likely to hit a crash.

#### How to fix it

Add a short note to the README table row, e.g. append "*(⚠️ deprecated — see CHANGELOG, pending MJX migration)*" to the mabrax row's summary text. One line, no code risk.

---

## PART 3 — CODE QUALITY ENHANCEMENTS

### Enhancement 1: `jaxmarl/registration.py`'s dispatch logic is a fragile, easy-to-break if/elif chain

**File:** `jaxmarl/registration.py:95-164`

#### Background

A common way to map a string name to a constructor is a long chain of `if name == "a": ... elif name == "b": ...`. This works, but it's fragile: Python's `elif` only continues the chain if it's *textually and structurally* attached to the preceding `if`. In this file, the dispatch isn't actually one chain — it's three separate chains stitched together: a `if env_id in (mabrax ids):` block, then `elif env_id == "storm":` *continues from that same `if`* even though mabrax and storm are unrelated, then later a brand new `if env_id.startswith("JaxRobotarium"):` starts a second, independent chain. This is exactly the kind of structure that produced Bug 2 above — a one-character typo inside a long `elif` chain silently produces an unassigned variable instead of a clear error, and nothing in the structure of the code stops a future contributor from making the same mistake when adding environment #39.

#### How to fix it

Replace the `if`/`elif` chain with an explicit lookup table — a Python `dict` mapping each string ID to a small constructor function (e.g. a `lambda **kw: ArcticTransport(**kw)`), then `make()` becomes:

```python
_REGISTRY = {
    "ant_4x2": lambda **kw: Ant(**kw),
    "storm": lambda **kw: InTheGrid(**kw),
    ...
    "JaxRobotarium_arctic_transport": lambda **kw: ArcticTransport(**kw),
    ...
}

def make(env_id, **env_kwargs):
    if env_id not in _REGISTRY:
        raise ValueError(f"Unknown environment id: {env_id!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[env_id](**env_kwargs)
```

This has two benefits: (1) a typo'd key in the dict is still a bug, but now `env_id not in _REGISTRY` gives a clear, helpful error message instead of a cryptic `UnboundLocalError`; (2) it's impossible to accidentally nest an unrelated environment's dispatch inside another one's `if`/`elif` chain, because there's no chain at all.

### Enhancement 2: Mutable default argument in `GymnaxToJaxMARL.__init__`

**File:** `jaxmarl/wrappers/gymnax.py:14`

#### Background you need

In Python, default argument values are evaluated **once**, when the function is defined — not once per call. If that default value is a mutable object like a `dict` or `list`, every call that doesn't pass its own value ends up sharing the *same* object. If any code path ever mutates that shared default in place (e.g. `env_kwargs["foo"] = 1`), every future call that also relies on the default will see that leftover value, because they're all literally the same dict in memory. This is one of Python's most famous footguns.

#### What's there

```python
def __init__(self, env_name: str, env_kwargs: dict = {}):
    self.env_name = env_name
    self._env, self.env_params = gymnax.make(env_name, **env_kwargs)
```

Right now, `env_kwargs` is only ever read (`**env_kwargs`), never mutated in place inside this method, so there's no live bug today. But it's a trap for the next person who adds a line like `env_kwargs["seed"] = 0` before the `gymnax.make(...)` call — thinking they're setting a local default — and ends up silently corrupting every other instance of `GymnaxToJaxMARL` constructed afterward without an explicit `env_kwargs`.

#### How to fix it

```python
def __init__(self, env_name: str, env_kwargs: dict | None = None):
    env_kwargs = env_kwargs if env_kwargs is not None else {}
    self.env_name = env_name
    self._env, self.env_params = gymnax.make(env_name, **env_kwargs)
```

Now every call that doesn't pass `env_kwargs` gets a fresh, independent empty dict.

### Enhancement 3: Wrappers detect environment type by guessing from the class name string

**Files:** `jaxmarl/wrappers/baselines.py:380-395`, `jaxmarl/wrappers/transformers.py:18-25`

#### Background

Both files contain logic shaped like:

```python
if "smax" in env.name.lower():
    ...
elif "overcooked" in env.name.lower():
    ...
elif "hanabi" in env.name.lower():
    ...
```

This is deciding *what kind of environment this is* by checking whether a specific substring appears in its `.name` attribute, rather than checking what Python class it actually is.

#### Why this is fragile

If a user subclasses `SMAX` to build their own variant and names it, say, `class MyCombatEnv(SMAX): ...`, then `env.name` would be `"MyCombatEnv"`, which doesn't contain `"smax"` — so none of these branches fire, and the code silently falls through to whatever the default/fallback behavior is (or, in the `transformers.py` case, raises `NotImplementedError`) even though the environment genuinely *is* a SMAX environment underneath. The failure is silent and hard to diagnose, because nothing about the user's subclass looks wrong at a glance.

#### How to fix it

Use `isinstance(env, SMAX)` (or whatever the real base class is) instead of string matching on the name. If there's a reason the actual class can't be imported/checked directly in that file (e.g. to avoid a circular import), an alternative is to give each environment class an explicit `env_type: ClassVar[str] = "smax"` attribute and check that — still far more robust than substring-matching a free-form display name.

### Enhancement 4: QMIX/VDN/IQL RNN scripts duplicate huge amounts of logic, which is exactly how Bug 1 ended up in two files at once

**Files:** `baselines/QLearning/qmix_rnn.py`, `vdn_rnn.py` (and likely `transf_qmix.py`, `shaq.py`, `iql_rnn.py`)

#### What's there

The `ScannedRNN` module definition, the rollout-collection loop (`_step_env`), the replay buffer setup, and the TD-target computation are copy-pasted nearly verbatim across these files. That's *why* Bug 1 exists identically in both `qmix_rnn.py` and `vdn_rnn.py` — it was written once and then copied, bug included.

#### How to fix it

Extract the shared, non-algorithm-specific pieces (the `ScannedRNN` class, the rollout-collection scan, the buffer construction/sampling boilerplate) into a shared `baselines/QLearning/utils.py` (or similar) module that each script imports from. Keep each script's actual *algorithm-specific* math (e.g. QMIX's mixing network vs. VDN's plain sum) as the only thing that differs between files. This way, fixing Bug 1 once in the shared module fixes it everywhere, instead of requiring someone to separately hunt down and patch every near-duplicate copy.

### Enhancement 5: QMIX's target network update is an unusual, under-documented hybrid

**File:** `baselines/QLearning/qmix_rnn.py:531-543`

#### Background you need

In Q-learning, a "target network" is a slow-moving, delayed copy of the main network, used to compute TD targets so training doesn't chase its own rapidly-moving predictions (which causes instability). There are two standard ways to update it: a **hard update** (every `N` steps, just copy the main network's weights over exactly) or a **soft/Polyak update** (every step, nudge the target slightly toward the main network: `target = tau * main + (1 - tau) * target`, with a small `tau` like 0.005).

#### What's there

```python
qmix_rnn.py:531-543
# (periodically, every TARGET_UPDATE_INTERVAL steps:)
target_network_params = optax.incremental_update(main_params, target_network_params, TAU)
```

This does a **soft** update, but only **periodically** (every `TARGET_UPDATE_INTERVAL` steps), instead of either a hard copy every `N` steps *or* a soft nudge every single step. This hybrid is unusual and easy to misconfigure: someone who sets `TAU=1.0` expecting that to behave like "hard update every interval" is correct, but someone who sets a small `TAU` (expecting the usual soft-update meaning) combined with a long `TARGET_UPDATE_INTERVAL` will find the target network barely moves at all, because the small nudge only happens rarely instead of every step.

#### How to fix it

Not a correctness bug — the math is internally consistent — but it deserves a clarifying comment next to the config fields explaining this specific interaction, and/or a config-time validation/warning if `TAU < 1.0` and `TARGET_UPDATE_INTERVAL` is large, since that combination is very likely a misconfiguration rather than an intentional choice.

### Enhancement 6: The `F821` lint rule (catches undefined-name typos) is turned off for the largest parts of the codebase

**File:** `pyproject.toml:108-110`

#### Background you need

`ruff` is a Python linter. One of its checks, `F821` ("undefined name"), flags code that references a variable which was never defined — exactly the class of mistake behind Bug 2 above (`env` never assigned because no branch matched). It's one of the few lint rules that can catch a real `NameError`-causing bug before the code even runs.

#### What's there

```toml
[tool.ruff.lint.per-file-ignores]
"jaxmarl/environments/**" = [..., "F821", ...]
"jaxmarl/gridworld/**"    = [..., "F821", ...]
"jaxmarl/viz/**"          = [..., "F821", ...]
```

`F821` is explicitly disabled for `jaxmarl/environments/**`, `jaxmarl/gridworld/**`, and `jaxmarl/viz/**` — which together make up the majority of the library's actual simulation code. The comment above this block ("older environment implementations not yet updated to 2.0 style") suggests this was a pragmatic, temporary call made at some point to avoid a wall of pre-existing lint failures, not a permanent design decision.

#### How to fix it

This isn't a one-line fix — turning `F821` back on everywhere at once would likely surface a batch of pre-existing lint failures that would need to be triaged individually. A practical path: enable `F821` for one subdirectory at a time (e.g. start with `jaxmarl/viz/**`, which is smaller), fix whatever it flags, and repeat, rather than attempting all three directories in one pass.

### Enhancement 7: Small dead-code cleanups

Three unrelated, low-effort tidy-ups found during the audit:

- **`jaxmarl/environments/coin_game/coin_game.py:573-595`** — the `if __name__ == "__main__":` block at the bottom of the file is almost entirely commented-out example code (a rollout + gif-saving demo). Either delete it, or uncomment it and verify it still runs, since as-is it's confusing dead weight that looks like it's supposed to do something.
- **`jaxmarl/environments/switch_riddle/switch_riddle.py:47`** — `self.a_to_i = {a: i for i, a in enumerate(self.agents)}` is built in the constructor but never read anywhere else in the class. Either remove it, or use it (e.g. in `render()`, to turn an agent index back into a human-readable agent name).
- **`jaxmarl/environments/spaces.py:32`** — a comment on the `Discrete` space says it "can't compose for multi-discrete" as if that were a current limitation, but a `MultiDiscrete` class already exists further down in the very same file. The comment looks like a stale leftover copied from an upstream library (gymnax) before `MultiDiscrete` was added here, and should be removed or updated.
- **`mkdocs.yml:3`** — `repo_url: https://github.com/flairox/jaxmarl` uses all-lowercase, while the real GitHub org is `FLAIROx` (mixed case) and every other reference in the repo uses the correct casing. GitHub URLs happen to redirect case-insensitively so nothing is actually broken, but it's inconsistent and worth matching for clarity.

---

## PART 4 — NEW FEATURE IDEAS

Each of these explains, from scratch, what the feature is and why it would concretely help someone using this library — not just "it would be nice."

### Feature 1: Close the baseline test gap — only 4 of ~26 algorithm scripts have any test, and even those don't check learning quality

**Where:** `tests/baselines/` currently has `test_ippo_hanabi.py`, `test_mappo_smax.py`, `test_pqn_overcooked.py`, `test_qmix_mpe.py`. `baselines/IPPO/`, `baselines/MAPPO/`, `baselines/QLearning/` together contain roughly 26 individual algorithm scripts.

#### What's missing, concretely

Scripts like `ippo_rnn_mabrax`, `ippo_ff_switch_riddle`, `mappo_rnn_hanabi`, `iql_rnn`, `vdn_rnn`, `qmix_rnn`, `shaq.py`, `transf_qmix.py`, and more have **zero automated tests** — nobody would notice if a change broke them until a human happened to run them manually. And the 4 tests that do exist only check that the training script runs to completion without crashing (`assert returncode == 0`) — they don't check that the agent actually *learned* anything. This is exactly why Bug 1 above could exist undetected: the buggy script runs fine, produces a trained model file, and exits with code 0 — it just learns a worse policy than it should, and nothing in CI measures "how good is the policy," only "did the process die."

#### Why this matters / how it helps

Two separate, complementary improvements:
1. **Crash-smoke tests for every untested script** — a cheap test (run the script for a tiny number of steps, assert it exits cleanly and produces no `NaN`s) catches import errors, shape mismatches, and config drift for the ~22 currently-unprotected scripts, mirroring what the existing 4 tests already do.
2. **A learning-quality regression check for at least a couple of key scripts** — e.g. "after training IPPO on a trivial scenario for a fixed, small budget, the average episodic return must exceed some threshold X" (where X is picked conservatively, well below what a correct implementation reliably achieves). This is the only kind of test that would have caught Bug 1: a logic bug that doesn't crash anything, it just quietly makes the agent worse at learning.

### Feature 2: S5 RNN architecture

**Where this is already requested:** `CONTRIBUTING.md:35-39` — the project's own roadmap section lists:
```
- [x] improved RNN implementations. In the current implementation, the hidden size is dependent on "NUM_STEPS", it should be made independent.
- [ ] S5 RNN architecture.
```

#### What S5 is, from scratch

The RNNs currently used in this repo's baselines (e.g. `ScannedRNN`, built on a GRU-style cell) process a sequence of observations one time step at a time, carrying a hidden-state vector forward. This is simple and well-understood, but it has two known weaknesses: it's slow to train because the time steps can't be processed in parallel (each one depends on the previous one's output), and it can struggle to retain information over very long sequences. **S5** ("Simplified State Space Sequence model") is a more modern sequence-modeling architecture (from the state-space-model family, related to S4/Mamba-style models) that can be computed with much more parallelism across the time dimension while still being a genuine recurrent/sequential model suitable for the same partially-observable RL settings this repo targets.

#### Why it would help

Multi-agent RL training in this repo already relies heavily on JAX's ability to parallelize computation on GPU/TPU (that's the entire premise of the library — "GPU-enabled efficiency" per the README). A sequence architecture that parallelizes better across time steps would plug directly into that existing strength, potentially speeding up training of any algorithm that currently uses an RNN (IPPO-RNN, MAPPO-RNN, QMIX-RNN, VDN-RNN, SHAQ, TransfQMIX), and because the maintainers have already listed it as wanted, a contribution here wouldn't need the usual "is this the right fit for the project" back-and-forth that new environments/algorithms typically require per `CONTRIBUTING.md`.

### Feature 3: A Gymnasium/PettingZoo-compatible "outward" wrapper

**Where:** `jaxmarl/wrappers/gymnax.py` currently only has `GymnaxToJaxMARL` — which converts a single-agent `gymnax` environment *into* a JaxMARL multi-agent environment (useful mainly for debugging multi-agent algorithm code against a trivial single-agent problem). There is no wrapper in the **other direction**.

#### Background you need

**PettingZoo** and **Gymnasium** are the two most widely-used standard Python interfaces for RL environments in the broader (non-JAX) Python RL ecosystem — most general-purpose RL libraries (e.g. RLlib, Stable-Baselines3, SuperSuit-based tooling) expect an environment object that exposes a PettingZoo- or Gymnasium-shaped API (`reset()`, `step(actions)` returning specific tuple shapes, specific attribute names for observation/action spaces, etc.), not JaxMARL's own `MultiAgentEnv` interface.

The irony here: this repo's own README states that several of its MPE environments were explicitly ported *from* PettingZoo in the first place — the environments are conceptually compatible — but once converted into JaxMARL's native API, there's currently no way to hand one back to PettingZoo-based tooling.

#### Why it would help

A user who wants to, say, run a JaxMARL environment through an existing, battle-tested library like RLlib (for a baseline comparison, or because they want an algorithm JaxMARL doesn't implement) currently cannot do so without writing their own adapter from scratch. A standard outward-facing wrapper — e.g. `JaxMARLToPettingZoo(env)` that exposes `.reset()`, `.step()`, `.observation_space(agent)`, `.action_space(agent)` in the shape PettingZoo's `ParallelEnv` expects, converting JAX arrays to NumPy at the boundary — would let any of this repo's environments be dropped directly into the much larger existing ecosystem of multi-agent RL tooling, without requiring that tooling to understand JAX at all.

### Feature 4: Eval-only mode / checkpoint-resume for baseline scripts

**Where:** every script in `baselines/{IPPO,MAPPO,QLearning}/` already has a `SAVE_PATH` config option and `save_params(...)` call (e.g. `qmix_rnn.py:720-739`) that writes trained weights to disk using `safetensors`.

#### What's missing

Nothing currently *reads* those saved checkpoint files back in. Every script only supports two modes: `single_run` (train from scratch, optionally save at the end) or `tune` (a Weights & Biases hyperparameter sweep, also training from scratch each time). There is no way to:
- Load a previously-saved checkpoint and continue training it further (e.g. you trained for 1M steps yesterday, and want to train for 1M more today without starting over).
- Load a saved checkpoint purely to **evaluate** it — run some number of test episodes and report the average return — without spending any time or compute retraining it.

#### Why it would help

Training a MARL baseline from scratch, especially the longer SMAX/Hanabi configs (some configs run for `1e7`+ environment steps, per the `TOTAL_TIMESTEPS` config values), can take a meaningful amount of wall-clock time and compute. Right now, every time someone wants to just *look at* how a previously-trained agent behaves (e.g. to render a video, to check a specific metric, to compare against a new baseline), they either have to retrain it from scratch, or write ad-hoc one-off loading code themselves. An `eval_only: true` config flag (load `SAVE_PATH`'s checkpoint, skip training, run test rollouts, report/save results) and a `resume_from: <path>` option (load a checkpoint as the starting point instead of a fresh random initialization) are both small, additive features that reuse code that's already 90% there (the save path exists; it just needs a matching load path).

### Feature 5: `jaxmarl.list_environments()` with structured metadata

**Where:** `jaxmarl/registration.py`'s `registered_envs` is currently a flat Python list of roughly 38 raw string identifiers (`"MPE_simple_v3"`, `"storm"`, `"hanabi"`, `"ant_4x2"`, ...), with no attached information about what each one actually *is*.

#### Why this is a gap

If a new user wants to find, say, "all fully-cooperative environments" or "all environments with exactly 2 agents" or "all continuous-action environments," their only option today is reading through the README's environment table by hand and cross-referencing it against the registration list, or opening up each environment's source file individually. The information needed to answer those questions already implicitly exists (every environment class knows its own `num_agents`, and whether its action space is discrete or continuous, etc.) — it's just not exposed anywhere as queryable metadata.

#### How it would help

A function like:

```python
jaxmarl.list_environments(cooperative=True, num_agents=2)
# -> ["storm_2p", "coin_game", ...]
```

would let users (and other tools built on top of JaxMARL, like a future benchmark dashboard or automated test-matrix generator) discover and filter environments programmatically instead of relying on manually-maintained documentation tables staying perfectly in sync with the code (which, as this audit's own Cleanup 3 above shows, they don't always do).

### Feature 6: Track the MABrax → MJX migration as an actual GitHub issue

**Where:** currently only mentioned in `CHANGELOG.md:27-30` as a changelog note ("this environment needs migrating to MJX"), not as a trackable, linkable, assignable GitHub issue.

#### Background you need

**Brax** is a JAX-native physics engine that the `mabrax` environment is built on; **MJX** is Google DeepMind's newer JAX port of the MuJoCo physics engine, which is the direction the broader JAX/robotics ecosystem has been consolidating around. The old Brax physics backend this environment depends on is no longer actively maintained upstream, which is exactly why `mabrax`'s own test is skip-marked as crashing on current `jaxlib` versions (see Cleanup 3 above).

#### Why filing a real issue would help

A CHANGELOG entry is a historical record, not a task board — it doesn't show up in GitHub's issue list, can't be commented on, labeled, assigned to a contributor, or linked from a pull request that claims to fix it. Converting this into an actual GitHub issue (with the context already written up above: why Brax is a problem, what MJX would need to replace, and the fact that this is entirely blocking the one skip-marked test) turns a known-but-buried fact into something a contributor can actually pick up and claim.

### Feature 7: Small community-health additions

Two low-effort, standard open-source project additions that are missing:
- **`CODE_OF_CONDUCT.md`** — a standard file most mature open-source projects include; its absence is flagged automatically by GitHub's "community profile" checklist, and adding one is a well-understood, low-risk task (most projects adopt the Contributor Covenant template as-is).
- **CI test-coverage reporting** — `ci.yml`'s test job currently runs `pytest ./tests/ -v -x` with no coverage measurement (e.g. via `pytest-cov`) and no badge in the README showing what percentage of the codebase is actually exercised by tests. Given Feature 1 above (most baseline scripts have zero tests), a visible coverage number would make that gap impossible to overlook in the future, instead of requiring someone to manually audit it (as this document just did).

"""Tests for the IC3Net-family baselines in baselines/learned-communication.

The returns and the communication step are checked against NumPy ports of the
loops in the IC3Net reference implementation (https://github.com/IC3Net/IC3Net):
`Trainer.compute_grad` in trainer.py and `CommNetMLP.forward` in comm.py.
"""

import importlib.util
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

SCRIPT = os.path.join("baselines", "learned-communication", "ic3net.py")


def _load_ic3net():
    spec = importlib.util.spec_from_file_location("ic3net", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ic3net = _load_ic3net()


def _reference_returns(rewards, episode_masks, episode_mini_masks, gamma, mean_ratio):
    """The return computation of the reference's `Trainer.compute_grad`."""
    coop_returns = np.zeros_like(rewards)
    ncoop_returns = np.zeros_like(rewards)
    returns = np.zeros_like(rewards)
    prev_coop_return = 0
    prev_ncoop_return = 0
    for i in reversed(range(rewards.shape[0])):
        coop_returns[i] = rewards[i] + gamma * prev_coop_return * episode_masks[i]
        ncoop_returns[i] = (
            rewards[i]
            + gamma * prev_ncoop_return * episode_masks[i] * episode_mini_masks[i]
        )
        prev_coop_return = coop_returns[i].copy()
        prev_ncoop_return = ncoop_returns[i].copy()
        returns[i] = (mean_ratio * coop_returns[i].mean()) + (
            (1 - mean_ratio) * ncoop_returns[i]
        )
    return returns


@pytest.mark.parametrize("mean_ratio", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("gamma", [1.0, 0.9])
def test_returns_match_reference(mean_ratio, gamma):
    rng = np.random.default_rng(0)
    steps, envs, agents = 30, 4, 5
    reward = rng.normal(size=(steps, envs, agents))
    done = rng.random((steps, envs)) < 0.1
    completed = rng.random((steps, envs, agents)) < 0.2

    returns = ic3net.compute_returns(
        jnp.asarray(reward),
        jnp.asarray(done),
        jnp.asarray(completed),
        gamma,
        mean_ratio,
    )

    for e in range(envs):
        # The reference ends every episode at its last step, and only reads
        # is_completed on steps that don't end the episode.
        last = done[:, e] | (np.arange(steps) == steps - 1)
        episode_masks = np.where(last[:, None], 0.0, 1.0) * np.ones(agents)
        episode_mini_masks = np.where(last[:, None], 1.0, 1.0 - completed[:, e])
        expected = _reference_returns(
            reward[:, e], episode_masks, episode_mini_masks, gamma, mean_ratio
        )
        np.testing.assert_allclose(returns[:, e], expected, rtol=1e-5, atol=1e-5)


def _dense(p, x):
    return x @ np.asarray(p["kernel"]) + np.asarray(p.get("bias", 0.0))


def _lstm(p, x, h, c):
    sigmoid = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
    i = sigmoid(_dense(p["ii"], x) + _dense(p["hi"], h))
    f = sigmoid(_dense(p["if"], x) + _dense(p["hf"], h))
    g = np.tanh(_dense(p["ig"], x) + _dense(p["hg"], h))
    o = sigmoid(_dense(p["io"], x) + _dense(p["ho"], h))
    c = f * c + i * g
    return o * np.tanh(c), c


def _reference_step(p, obs, h, c, alive, talk, communicate, gated, comm_passes, lstm):
    """One step of the reference's `CommNetMLP.forward` (recurrent) or `RNN.forward`."""
    n, hid_size = h.shape
    x = _dense(p["encoder"], obs)
    if not communicate and lstm:
        h, c = _lstm(p["lstm"], x, h, c)
    elif not communicate:
        h = np.tanh(_dense(p["recurrent"], h) + x)
    else:
        comm_mask = np.ones((n, n)) - np.eye(n)
        num_agents_alive = alive.sum()
        agent_mask = np.broadcast_to(alive.reshape(1, 1, n), (1, n, n))[..., None]
        if gated:
            agent_mask = (
                agent_mask
                * np.broadcast_to(talk.reshape(1, 1, n), (1, n, n))[..., None]
            )
        agent_mask_transpose = agent_mask.transpose(0, 2, 1, 3)
        for i in range(comm_passes):
            comm = np.broadcast_to(h.reshape(1, n, 1, hid_size), (1, n, n, hid_size))
            comm = comm * comm_mask.reshape(1, n, n, 1)
            if num_agents_alive > 1:
                comm = comm / (num_agents_alive - 1)
            comm = comm * agent_mask
            comm = comm * agent_mask_transpose
            comm_sum = comm.sum(axis=1)[0]
            h, c = _lstm(p["lstm"], x + _dense(p[f"comm_{i}"], comm_sum), h, c)
    logits = _dense(p["Dense_0"], h)
    value = _dense(p["Dense_1"], h)[:, 0]
    talk_logits = _dense(p["talk"], h) if gated else None
    return h, c, logits, talk_logits, value


@pytest.mark.parametrize(
    "baseline, lstm",
    [("ic", False), ("ic", True), ("commnet", False), ("ic3net", False)],
)
@pytest.mark.parametrize("comm_passes", [1, 2])
def test_policy_step_matches_reference(baseline, lstm, comm_passes):
    communicate, gated, _ = ic3net.BASELINES[baseline]
    n, obs_dim, hidden, actions = 5, 7, 16, 4
    network = ic3net.CommPolicy(
        action_dim=actions,
        hidden_size=hidden,
        communicate=communicate,
        gated=gated,
        comm_passes=comm_passes,
        lstm=lstm,
    )
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    obs = jax.random.normal(keys[0], (n, obs_dim))
    h = jax.random.normal(keys[1], (n, hidden))
    c = jax.random.normal(keys[2], (n, hidden))
    params = network.init(keys[3], (h, c), obs, jnp.ones(n), jnp.ones(n))
    p = jax.tree.map(np.asarray, params["params"])

    # GPUs otherwise multiply float32 matrices in TF32.
    with jax.default_matmul_precision("float32"):
        # All alive and talking, some dead, some silent, and a single agent alive.
        for alive, talk in [
            ([1, 1, 1, 1, 1], [1, 1, 1, 1, 1]),
            ([1, 0, 1, 1, 0], [1, 1, 1, 1, 1]),
            ([1, 1, 1, 1, 1], [0, 1, 1, 0, 1]),
            ([1, 0, 1, 1, 0], [1, 1, 0, 1, 1]),
            ([0, 0, 1, 0, 0], [1, 1, 1, 1, 1]),
        ]:
            alive, talk = np.array(alive, float), np.array(talk, float)
            (h1, c1), logits, talk_logits, value = network.apply(
                params, (h, c), obs, jnp.asarray(alive), jnp.asarray(talk)
            )
            ref = _reference_step(
                p,
                np.asarray(obs),
                np.asarray(h),
                np.asarray(c),
                alive,
                talk,
                communicate,
                gated,
                comm_passes,
                lstm,
            )
            np.testing.assert_allclose(h1, ref[0], rtol=1e-4, atol=1e-5)
            if communicate or lstm:
                np.testing.assert_allclose(c1, ref[1], rtol=1e-4, atol=1e-5)
            np.testing.assert_allclose(logits, ref[2], rtol=1e-4, atol=1e-5)
            np.testing.assert_allclose(value, ref[4], rtol=1e-4, atol=1e-5)
            if gated:
                np.testing.assert_allclose(talk_logits, ref[3], rtol=1e-4, atol=1e-5)


def test_silent_agents_neither_send_nor_receive():
    network = ic3net.CommPolicy(
        action_dim=3, hidden_size=8, communicate=True, gated=True, comm_passes=1
    )
    keys = jax.random.split(jax.random.PRNGKey(1), 3)
    n = 4
    obs = jax.random.normal(keys[0], (n, 6))
    h = jax.random.normal(keys[1], (n, 8))
    carry = (h, jnp.zeros((n, 8)))
    params = network.init(keys[2], carry, obs, jnp.ones(n), jnp.ones(n))

    def hidden(talk, h=h):
        (h1, _), *_ = network.apply(params, (h, carry[1]), obs, jnp.ones(n), talk)
        return h1

    silent = jnp.zeros(n)
    talk = jnp.array([1.0, 0.0, 1.0, 1.0])
    # Agent 1 is silent: it receives nothing, and changing its hidden state
    # doesn't change what the others receive.
    np.testing.assert_allclose(hidden(talk)[1], hidden(silent)[1], rtol=1e-6)
    h_changed = h.at[1].add(1.0)
    np.testing.assert_allclose(
        hidden(talk, h_changed)[jnp.array([0, 2, 3])],
        hidden(talk)[jnp.array([0, 2, 3])],
        rtol=1e-6,
    )
    assert not np.allclose(hidden(talk)[0], hidden(silent)[0])


@pytest.mark.parametrize("baseline", ["ic", "iric", "commnet", "ic3net"])
def test_script_runs(baseline):
    result = subprocess.run(
        [
            sys.executable,
            SCRIPT,
            "+alg=ic3net_traffic_junction_easy",
            f"BASELINE={baseline}",
            "NUM_SEEDS=2",
            "alg.NUM_ENVS=4",
            "alg.TOTAL_TIMESTEPS=400",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "success" in result.stdout


def _config(baseline, **overrides):
    config = {
        "BASELINE": baseline,
        "MEAN_RATIO": None,
        "ENV_NAME": "traffic_junction_grid",
        "ENV_KWARGS": {"difficulty": "easy"},
        "NUM_ENVS": 3,
        "NUM_STEPS": 25,
        "TOTAL_TIMESTEPS": 3 * 25 * 2,
        "HIDDEN_SIZE": 8,
        "COMM_PASSES": 1,
        "COMM_ACTION_ONE": False,
        "LR": 0.001,
        "GAMMA": 1.0,
        "VALUE_COEF": 0.01,
        "ENT_COEF": 0.0,
        "DETACH_GAP": 3,
        "WANDB_MODE": "disabled",
    }
    config.update(overrides)
    return config


@pytest.mark.parametrize(
    "baseline, comm_action_one",
    [("ic", False), ("commnet", False), ("ic3net", False), ("ic3net", True)],
)
def test_rollout_feeds_the_previous_step_into_communication(baseline, comm_action_one):
    # Episodes of easy Traffic Junction last 20 of the 25 steps.
    learner = ic3net.make_learner(_config(baseline, COMM_ACTION_ONE=comm_action_one))
    params = learner.init_params(jax.random.PRNGKey(0))
    traj = jax.tree.map(np.asarray, learner.rollout(params, jax.random.PRNGKey(1)))

    # As in the reference, every agent counts as alive and none talks at the
    # first step; after that, alive comes from the previous step's info.
    assert np.all(traj.alive[0] == 1) and np.all(traj.talk[0] == 0)
    np.testing.assert_array_equal(traj.alive[1:], traj.alive_after[:-1])
    assert np.any(traj.alive_after == 0)  # cars off the road are not alive
    # An IC3Net agent's choice to talk takes effect at the next step.
    if learner.always_talk:
        assert np.all(traj.talk[1:] == 1)
    elif learner.gated:
        np.testing.assert_array_equal(traj.talk[1:], traj.talk_action[:-1])
        assert 0 < traj.talk_action.mean() < 1
    # Only the first episode counts.
    assert np.all(traj.valid[:20]) and not np.any(traj.valid[20:])
    assert np.all(traj.done[19]) and not np.any(traj.done[:19])


def _reference_loss(network, params, traj, returns, gated, value_coef, detach_gap):
    """The reference's loss (`Trainer.get_episode` and `compute_grad`), one
    environment and one step at a time."""
    actor, value_loss, steps = 0.0, 0.0, 0
    for e in range(traj.obs.shape[1]):
        hidden = tuple(jnp.zeros((1, *traj.obs.shape[2:3], 8)) for _ in range(2))
        log_probs, values = [], []
        for t in range(traj.obs.shape[0]):
            if not traj.valid[t, e]:
                break
            hidden, logits, talk_logits, value = network.apply(
                params,
                hidden,
                traj.obs[t, e][None],
                traj.alive[t, e][None],
                traj.talk[t, e][None],
            )
            log_prob = jnp.take_along_axis(
                jax.nn.log_softmax(logits[0]), traj.action[t, e][:, None], -1
            )[:, 0]
            if gated:
                log_prob = (
                    log_prob
                    + jnp.take_along_axis(
                        jax.nn.log_softmax(talk_logits[0]),
                        traj.talk_action[t, e][:, None],
                        -1,
                    )[:, 0]
                )
            log_probs.append(log_prob)
            values.append(value[0])
            if (t + 1) % detach_gap == 0:
                hidden = jax.lax.stop_gradient(hidden)
        n = len(values)
        log_probs, values = jnp.stack(log_probs), jnp.stack(values)
        alive_mask = traj.alive_after[:n, e]
        advantages = returns[:n, e] - jax.lax.stop_gradient(values)
        actor = actor - (advantages * log_probs * alive_mask).sum()
        value_loss = value_loss + (((values - returns[:n, e]) ** 2) * alive_mask).sum()
        steps += n
    return (actor + value_coef * value_loss) / steps


@pytest.mark.parametrize(
    "baseline, lstm",
    [("ic", False), ("ic", True), ("commnet", False), ("ic3net", False)],
)
def test_loss_and_gradients_match_reference(baseline, lstm):
    config = _config(baseline, NUM_STEPS=9, LSTM=lstm)
    learner = ic3net.make_learner(config)
    params = learner.init_params(jax.random.PRNGKey(0))
    rng = np.random.default_rng(0)
    steps, envs, agents, obs_dim = (
        9,
        3,
        5,
        params["params"]["encoder"]["kernel"].shape[0],
    )
    alive_after = (rng.random((steps, envs, agents)) < 0.7).astype(np.float32)
    done = np.zeros((steps, envs), dtype=bool)
    done[5, 0] = True  # the first environment's episode ends early
    valid = np.concatenate(
        [np.ones((1, envs), bool), ~np.cumsum(done, 0)[:-1].astype(bool)]
    )
    traj = ic3net.Transition(
        obs=rng.normal(size=(steps, envs, agents, obs_dim)).astype(np.float32),
        alive=np.concatenate(
            [np.ones((1, envs, agents), np.float32), alive_after[:-1]]
        ),
        talk=np.concatenate(
            [np.zeros((1, envs, agents)), rng.integers(0, 2, (steps - 1, envs, agents))]
        ).astype(np.float32),
        action=rng.integers(0, 2, (steps, envs, agents)),
        talk_action=rng.integers(0, 2, (steps, envs, agents)),
        reward=rng.normal(size=(steps, envs, agents)).astype(np.float32),
        alive_after=alive_after,
        completed=rng.random((steps, envs, agents)) < 0.1,
        valid=valid,
        done=done,
        success=valid,
    )
    traj = jax.tree.map(jnp.asarray, traj)

    # The returns come from the trajectory's own done and completed.
    returns = learner.returns(traj)
    _, _, mean_ratio = ic3net.BASELINES[baseline]
    for e in range(envs):
        last = np.asarray(done[:, e]) | (np.arange(steps) == steps - 1)
        expected = _reference_returns(
            np.asarray(traj.reward[:, e]),
            np.where(last[:, None], 0.0, 1.0) * np.ones(agents),
            np.where(last[:, None], 1.0, 1.0 - np.asarray(traj.completed[:, e])),
            config["GAMMA"],
            mean_ratio,
        )
        np.testing.assert_allclose(returns[:, e], expected, rtol=1e-5, atol=1e-5)

    def port(params):
        return learner.loss(params, traj, returns)[0]

    def reference(params):
        return _reference_loss(
            learner.network,
            params,
            traj,
            returns,
            learner.gated,
            config["VALUE_COEF"],
            config["DETACH_GAP"],
        )

    with jax.default_matmul_precision("float32"):
        loss, grads = jax.value_and_grad(port)(params)
        ref_loss, ref_grads = jax.value_and_grad(reference)(params)
    np.testing.assert_allclose(loss, ref_loss, rtol=1e-5)
    jax.tree.map(
        lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-6),
        grads,
        ref_grads,
    )


def test_optimizer_matches_pytorch_rmsprop():
    # torch.optim.RMSprop(alpha=0.97, eps=1e-6): v = alpha v + (1 - alpha) g^2,
    # p -= lr g / (sqrt(v) + eps).
    lr, alpha, eps = 0.001, 0.97, 1e-6
    rng = np.random.default_rng(0)
    params = {"w": jnp.asarray(rng.normal(size=6))}
    tx = ic3net.make_optimizer(lr)
    state = tx.init(params)
    p, v = np.asarray(params["w"], np.float64), np.zeros(6)
    for scale in [1e-6, 1e-5, 1e-3]:
        g = rng.normal(size=6) * scale
        updates, state = tx.update({"w": jnp.asarray(g, jnp.float32)}, state, params)
        params = optax.apply_updates(params, updates)
        v = alpha * v + (1 - alpha) * g**2
        p = p - lr * g / (np.sqrt(v) + eps)
        np.testing.assert_allclose(params["w"], p, rtol=1e-6, atol=1e-8)


def test_script_runs_on_predator_prey():
    result = subprocess.run(
        [
            sys.executable,
            SCRIPT,
            "+alg=ic3net_predator_prey_easy",
            "BASELINE=ic3net",
            "NUM_SEEDS=2",
            "SEED_CHUNK=1",
            "alg.NUM_ENVS=4",
            "alg.TOTAL_TIMESTEPS=400",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "returned_episode_lengths" in result.stdout

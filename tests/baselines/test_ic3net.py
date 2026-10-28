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


def _reference_step(p, obs, h, c, alive, talk, communicate, gated, comm_passes):
    """One step of the reference's `CommNetMLP.forward` (recurrent) or `RNN.forward`."""
    n, hid_size = h.shape
    x = _dense(p["encoder"], obs)
    if not communicate:
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


@pytest.mark.parametrize("baseline", ["ic", "commnet", "ic3net"])
@pytest.mark.parametrize("comm_passes", [1, 2])
def test_policy_step_matches_reference(baseline, comm_passes):
    communicate, gated, _ = ic3net.BASELINES[baseline]
    n, obs_dim, hidden, actions = 5, 7, 16, 4
    network = ic3net.CommPolicy(
        action_dim=actions,
        hidden_size=hidden,
        communicate=communicate,
        gated=gated,
        comm_passes=comm_passes,
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
            )
            np.testing.assert_allclose(h1, ref[0], rtol=1e-4, atol=1e-5)
            if communicate:
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


def test_curriculum_raises_add_rate():
    config = {
        "BASELINE": "ic",
        "MEAN_RATIO": None,
        "ENV_NAME": "traffic_junction_grid",
        "ENV_KWARGS": {"difficulty": "easy"},
        "NUM_ENVS": 2,
        "NUM_STEPS": 20,
        "TOTAL_TIMESTEPS": 2 * 20 * 80,
        "HIDDEN_SIZE": 8,
        "COMM_PASSES": 1,
        "LR": 0.001,
        "GAMMA": 1.0,
        "VALUE_COEF": 0.01,
        "ENT_COEF": 0.0,
        "DETACH_GAP": 10,
        "WANDB_MODE": "disabled",
        "CURRICULUM": {
            "KEY": "add_rate",
            "START": 0.1,
            "END": 0.3,
            "STEP": 0.01,
            "BEGIN": 0.25,
            "FINISH": 0.75,
        },
    }
    out = jax.jit(ic3net.make_train(config))(jax.random.PRNGKey(0))
    add_rate = np.asarray(out["metrics"]["add_rate"])
    np.testing.assert_allclose(add_rate[:20], 0.1)
    np.testing.assert_allclose(add_rate[60:], 0.3)
    steps = np.diff(add_rate)
    assert np.all(steps >= 0) and np.all(steps <= 0.01 + 1e-6)

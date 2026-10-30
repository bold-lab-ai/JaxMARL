"""
Tests for the grid Predator-Prey environment, including step-by-step
correspondence with the IC3Net reference implementation (ic3net_predator_prey.py).

Details of the reference that the README leaves out:

- The reference declares its observation space as (vocab, 2v+1, 2v+1) with
  high=1, but returns (2v+1, 2v+1, vocab) arrays with counts up to num_agents.
- With vision at least dim, the reference refuses moves right and down, because
  it checks them against its padded grid; here predators move normally.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ic3net_predator_prey import PredatorPreyEnv

from jaxmarl import make
from jaxmarl.environments.predator_prey_grid import PredatorPreyGrid

# The paper's easy, medium and hard settings.
CONFIGS = [
    dict(num_agents=3, dim=5, vision=0, max_steps=20),
    dict(num_agents=5, dim=10, vision=1, max_steps=40),
    dict(num_agents=10, dim=20, vision=1, max_steps=80),
]
MODES = ["cooperative", "competitive", "mixed"]


def _reference(env: PredatorPreyGrid) -> PredatorPreyEnv:
    ref = PredatorPreyEnv()
    ref.multi_agent_init(
        SimpleNamespace(
            dim=env.dim,
            vision=env.vision,
            moving_prey=False,
            mode=env.mode,
            enemy_comm=False,
            nenemies=1,
            nfriendly=env.num_agents,
            no_stay=not env.stay,
        )
    )
    return ref


def _sync_reset(ref: PredatorPreyEnv, state):
    """Reset the reference to the JAX env's initial positions."""
    ref.reset()
    ref.predator_loc = np.array(state.predator_loc)
    ref.prey_loc = np.array(state.prey_loc)[None]
    return ref._get_obs()


def _random_actions(env, key):
    keys = jax.random.split(key, env.num_agents)
    return {a: env.action_space(a).sample(keys[i]) for i, a in enumerate(env.agents)}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("config", CONFIGS)
def test_matches_reference(config, mode):
    """Observations, rewards, termination and success match IC3Net step by step."""
    env = PredatorPreyGrid(mode=mode, **config)
    ref = _reference(env)
    step_env = jax.jit(env.step_env)
    key = jax.random.PRNGKey(0)

    for _ in range(3):
        key, key_reset = jax.random.split(key)
        obs, state = env.reset(key_reset)
        ref_obs = _sync_reset(ref, state)
        for t in range(env.max_steps):
            np.testing.assert_array_equal(
                np.stack([obs[a] for a in env.agents]),
                ref_obs.reshape(env.num_agents, -1),
            )
            key, key_act, key_step = jax.random.split(key, 3)
            actions = _random_actions(env, key_act)
            obs, state, reward, done, info = step_env(key_step, state, actions)
            ref_obs, ref_reward, ref_over, _ = ref.step(
                [int(actions[a]) for a in env.agents]
            )

            np.testing.assert_allclose(
                [reward[a] for a in env.agents], ref_reward, rtol=1e-6
            )
            np.testing.assert_array_equal(state.reached_prey, ref.reached_prey == 1)
            if mode != "competitive":
                assert bool(info["success"][0]) == bool(ref.stat["success"])
            # The reference leaves the time limit to the training loop.
            assert bool(done["__all__"]) == (ref_over or t + 1 == env.max_steps)
            if ref_over:
                break


def test_episode_lasts_max_steps():
    env = make("predator_prey_grid", mode="cooperative", max_steps=7)
    obs, state = env.reset(jax.random.PRNGKey(0))
    stay = {a: jnp.int32(4) for a in env.agents}
    for t in range(1, 8):
        obs, state, _, done, _ = env.step_env(jax.random.PRNGKey(t), state, stay)
        assert bool(done["__all__"]) == (t == 7)
        assert all(bool(done[a]) == (t == 7) for a in env.agents)


def _state_with(env, predator_loc, prey_loc):
    _, state = env.reset(jax.random.PRNGKey(0))
    return state.replace(
        predator_loc=jnp.array(predator_loc), prey_loc=jnp.array(prey_loc)
    )


@pytest.mark.parametrize(
    "mode, expected",
    [
        ("cooperative", [0.1, 0.1, -0.05]),
        ("competitive", [0.025, 0.025, -0.05]),
        ("mixed", [0.0, 0.0, -0.05]),
    ],
)
def test_rewards_per_mode(mode, expected):
    """Two of three predators step onto the prey."""
    env = PredatorPreyGrid(mode=mode)
    state = _state_with(env, [[1, 2], [3, 2], [4, 4]], [2, 2])
    actions = {"agent_0": 2, "agent_1": 0, "agent_2": 4}  # DOWN, UP, STAY
    _, state, reward, done, _ = env.step_env(jax.random.PRNGKey(0), state, actions)
    np.testing.assert_allclose([reward[a] for a in env.agents], expected, rtol=1e-6)
    assert not bool(done["__all__"])


def test_predators_on_prey_stop_and_mixed_ends_when_all_arrive():
    env = PredatorPreyGrid(mode="mixed")
    state = _state_with(env, [[2, 2], [2, 1], [0, 0]], [2, 2])
    state = state.replace(reached_prey=jnp.array([True, False, False]))
    right = {a: 1 for a in env.agents}
    _, state, _, done, _ = env.step_env(jax.random.PRNGKey(0), state, right)
    np.testing.assert_array_equal(state.predator_loc, [[2, 2], [2, 2], [0, 1]])
    assert not bool(done["__all__"])

    state = state.replace(predator_loc=jnp.array([[2, 2], [2, 2], [2, 1]]))
    _, state, _, done, info = env.step_env(jax.random.PRNGKey(1), state, right)
    assert bool(done["__all__"])
    assert bool(info["success"].all())


def test_predators_stay_inside_the_grid():
    env = PredatorPreyGrid(dim=5)
    state = _state_with(env, [[0, 0], [4, 4], [0, 4]], [2, 2])
    actions = {"agent_0": 0, "agent_1": 2, "agent_2": 1}  # UP, DOWN, RIGHT
    _, state, _, _, _ = env.step_env(jax.random.PRNGKey(0), state, actions)
    np.testing.assert_array_equal(state.predator_loc, [[0, 0], [4, 4], [0, 4]])


def test_reset_places_everyone_on_distinct_cells():
    env = PredatorPreyGrid(num_agents=10, dim=4)
    _, states = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(0), 256))
    locs = jnp.concatenate([states.predator_loc, states.prey_loc[:, None]], axis=1)
    cells = locs[..., 0] * env.dim + locs[..., 1]
    assert all(len(np.unique(c)) == env.num_agents + 1 for c in np.array(cells))
    assert bool(jnp.all((locs >= 0) & (locs < env.dim)))


def test_reset_places_prey_and_predators_uniformly():
    # The reference places every animal uniformly at random.
    env = PredatorPreyGrid(num_agents=3, dim=5)
    samples = 20000
    _, states = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(0), samples))
    for locs in (
        states.prey_loc,
        states.predator_loc[:, 0],
        states.predator_loc[:, -1],
    ):
        cells = np.asarray(locs[..., 0] * env.dim + locs[..., 1])
        share = np.bincount(cells, minlength=env.dim**2) / samples
        # The standard error of each share is about 0.0014.
        np.testing.assert_allclose(share, 1 / env.dim**2, atol=0.008)


def test_jit_vmap_and_auto_reset():
    env = make("predator_prey_grid", num_agents=5, dim=10, vision=1, max_steps=40)
    batch = 64
    obs, state = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(0), batch))

    def _step(carry, key):
        obs, state = carry
        key_act, key_step = jax.random.split(key)
        keys = jax.random.split(key_act, env.num_agents)
        actions = {
            a: jax.random.randint(keys[i], (batch,), 0, 5)
            for i, a in enumerate(env.agents)
        }
        obs, state, reward, done, info = jax.vmap(env.step)(
            jax.random.split(key_step, batch), state, actions
        )
        return (obs, state), done["__all__"]

    (obs, state), dones = jax.lax.scan(
        jax.jit(_step), (obs, state), jax.random.split(jax.random.PRNGKey(1), 100)
    )
    assert dones.shape == (100, batch)
    assert bool(jnp.all(state.step < env.max_steps))
    for a in env.agents:
        assert obs[a].shape == (batch, *env.observation_space(a).shape)


def test_no_stay_has_four_actions():
    env = PredatorPreyGrid(stay=False)
    assert all(env.action_space(a).n == 4 for a in env.agents)
    avail = env.get_avail_actions(env.reset(jax.random.PRNGKey(0))[1])
    assert all(avail[a].shape == (4,) for a in env.agents)


def test_invalid_arguments():
    with pytest.raises(ValueError):
        PredatorPreyGrid(mode="adversarial")
    with pytest.raises(ValueError):
        PredatorPreyGrid(num_agents=4, dim=2)


def test_accepts_actions_with_a_trailing_dimension():
    """The IPPO and MAPPO baselines pass each action as a (1,) array."""
    env = make("predator_prey_grid", num_agents=5, dim=10, vision=1, max_steps=40)
    _, state = env.reset(jax.random.PRNGKey(0))
    scalar = {a: jnp.int32(i % 2) for i, a in enumerate(env.agents)}
    vector = {a: v[None] for a, v in scalar.items()}
    obs_s, state_s, reward_s, _, _ = env.step_env(jax.random.PRNGKey(1), state, scalar)
    obs_v, state_v, reward_v, _, _ = env.step_env(jax.random.PRNGKey(1), state, vector)
    for a in env.agents:
        np.testing.assert_array_equal(obs_s[a], obs_v[a])
        assert float(reward_s[a]) == float(reward_v[a])
    jax.tree.map(np.testing.assert_array_equal, state_s, state_v)

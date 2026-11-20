"""
Tests for the grid Traffic Junction environment, including correspondence with the
IC3Net reference implementation (ic3net_traffic_junction.py).

Car arrivals are random, so the transition test replays each JAX step through the
reference: same pre-step state and actions, same arriving cars, then compares the
cars, rewards, collisions and observations.

Details of the reference and the paper that the README leaves out:

- The reference sizes the easy version's road ids from `dim` rather than the
  enlarged `dim + 1` grid, so one road cell, two below the junction, has the
  outside id. The tests apply the environment's per-cell ids to the reference.
- The reference counts cars that are off the road at cell (0, 0), which is never
  road and only falls inside a window from vision 6 on medium and 4 on hard.
- The reference's instructions give medium a maximum add_rate of 0.02, below its
  minimum of 0.05, which disables its curriculum; the default follows Table 4.
- The paper's text gives medium 20 cars, easy two routes per arrival point and
  medium two junctions; Table 4 and the reference give 10 cars and one of each,
  which the defaults follow.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ic3net_traffic_junction import TrafficJunctionEnv

from jaxmarl import make
from jaxmarl.environments.traffic_junction_grid import TrafficJunctionGrid
from jaxmarl.environments.traffic_junction_grid.traffic_junction_grid import (
    DIFFICULTIES,
)

GAS, BRAKE = 0, 1


class Reference(TrafficJunctionEnv):
    """The reference, with easy road ids counted over the enlarged grid.

    This is the port's one intended change to the reference: it sizes the easy
    vocabulary from ``dim`` rather than ``dim + 1``, so one road cell, two below
    the junction, has the outside id.
    """

    def multi_agent_init(self, args):
        super().multi_agent_init(args)
        if self.difficulty == "easy":
            base = sum(self.dims)
            self.OUTSIDE_CLASS += base - self.BASE
            self.CAR_CLASS += base - self.BASE
            self.BASE = base
            self.vocab_size = 1 + base + 1 + 1
            self._set_grid()
            self._set_paths_easy()


def _reference(env: TrafficJunctionGrid) -> Reference:
    ref = Reference()
    ref.multi_agent_init(
        SimpleNamespace(
            dim=env.dim,
            vision=env.vision,
            add_rate_min=env.add_rate,
            add_rate_max=env.add_rate,
            curr_start=0,
            curr_end=0,
            difficulty=env.difficulty,
            vocab_type="bool",
            nagents=env.num_agents,
        )
    )
    ref.reset()
    return ref


def _reference_routes(ref):
    return [np.array(path) for group in ref.routes for path in group]


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_routes_match_reference(difficulty):
    env = TrafficJunctionGrid(difficulty=difficulty)
    ref_routes = _reference_routes(_reference(env))
    assert (
        env.num_routes
        == len(ref_routes)
        == {"easy": 2, "medium": 12, "hard": 56}[difficulty]
    )
    for i, path in enumerate(ref_routes):
        assert int(env.route_lengths[i]) == len(path)
        np.testing.assert_array_equal(env.routes[i, : len(path)], path)


def _load_reference_state(ref, state, routes):
    alive = np.array(state.alive)
    ref.alive_mask = alive.astype(float)
    ref.wait = np.array(state.wait, dtype=float)
    ref.car_loc = np.array(state.car_loc, dtype=int)
    ref.car_last_act = np.array(state.last_action, dtype=int)
    ref.car_route_loc = np.array(state.route_pos)
    ref.route_id = [int(r) for r in state.route]
    ref.chosen_path = [routes[r] for r in ref.route_id]
    ref.cars_in_sys = int(alive.sum())
    ref.has_failed = int(state.has_failed)
    ref.is_completed = np.zeros(ref.ncar)


@pytest.mark.parametrize(
    "difficulty, vision",
    [("easy", 0), ("easy", 1), ("medium", 0), ("medium", 1), ("hard", 0)],
)
def test_transitions_match_reference(difficulty, vision):
    # A high arrival rate produces frequent arrivals, exits and collisions.
    env = TrafficJunctionGrid(difficulty=difficulty, vision=vision, add_rate=0.5)
    ref = _reference(env)
    routes = _reference_routes(ref)
    step_env = jax.jit(env.step_env)
    key = jax.random.PRNGKey(0)
    collisions = 0

    for _ in range(3):
        key, key_reset = jax.random.split(key)
        _, state = env.reset(key_reset)
        for _ in range(env.max_steps):
            key, key_act, key_step = jax.random.split(key, 3)
            # Mostly GAS so that cars reach the ends of their routes.
            acts = jax.random.bernoulli(key_act, 0.25, (env.num_agents,)).astype(int)
            actions = {a: acts[i] for i, a in enumerate(env.agents)}

            _load_reference_state(ref, state, routes)
            ref.is_completed = np.zeros(ref.ncar)
            for i, a in enumerate(np.array(acts)):
                ref._take_action(i, a)
            obs, next_state, reward, done, info = step_env(key_step, state, actions)

            # Arriving cars only take cars that are off the road, at most one per
            # arrival point, at the start of a route from that point.
            still_on_road = ref.alive_mask == 1
            assert bool(np.all(np.array(next_state.alive)[still_on_road]))
            arrived = np.array(next_state.alive) & ~still_on_road
            arrival_points = (
                np.array(next_state.route)[arrived] // env.routes_per_arrival
            )
            assert len(set(arrival_points)) == len(arrival_points)
            for j in np.nonzero(arrived)[0]:
                r = int(next_state.route[j])
                assert int(next_state.route_pos[j]) == 0
                ref.alive_mask[j] = 1
                ref.route_id[j] = r
                ref.chosen_path[j] = routes[r]
                ref.car_route_loc[j] = 0
                ref.car_loc[j] = routes[r][0]
                ref.cars_in_sys += 1

            ref_obs = ref._get_obs()
            ref_reward = ref._get_reward()
            np.testing.assert_array_equal(next_state.car_loc, ref.car_loc)
            np.testing.assert_array_equal(next_state.wait, ref.wait)
            np.testing.assert_array_equal(next_state.last_action, ref.car_last_act)
            np.testing.assert_allclose(
                [reward[a] for a in env.agents], ref_reward, rtol=1e-6, atol=1e-6
            )
            assert bool(next_state.has_failed) == bool(ref.has_failed)
            assert bool(info["success"][0]) == (not ref.has_failed)
            np.testing.assert_array_equal(info["alive"], ref.alive_mask == 1)
            np.testing.assert_array_equal(info["completed"], ref.is_completed == 1)
            for i, a in enumerate(env.agents):
                act, route, window = ref_obs[i]
                expected = np.concatenate([[act, route], np.ravel(window)])
                np.testing.assert_allclose(obs[a], expected, rtol=1e-6)

            collisions += int(np.sum(np.array(ref_reward) <= -10))
            state = next_state
    assert collisions > 0, "the test should exercise collisions"


def test_episode_lasts_max_steps():
    env = make("traffic_junction_grid", difficulty="medium", max_steps=9)
    _, state = env.reset(jax.random.PRNGKey(0))
    brake = {a: jnp.int32(BRAKE) for a in env.agents}
    for t in range(1, 10):
        _, state, _, done, _ = env.step_env(jax.random.PRNGKey(t), state, brake)
        assert bool(done["__all__"]) == (t == 9)


def test_arrival_rate_bounds():
    env = TrafficJunctionGrid(difficulty="hard", add_rate=0.0)
    _, state = env.reset(jax.random.PRNGKey(0))
    gas = {a: jnp.int32(GAS) for a in env.agents}
    for t in range(20):
        _, state, reward, _, _ = env.step_env(jax.random.PRNGKey(t), state, gas)
        assert not bool(state.alive.any())
        assert all(float(reward[a]) == 0.0 for a in env.agents)

    env = TrafficJunctionGrid(difficulty="hard", add_rate=1.0)
    _, state = env.reset(jax.random.PRNGKey(0))
    brake = {a: jnp.int32(BRAKE) for a in env.agents}
    _, state, _, _, _ = env.step_env(jax.random.PRNGKey(0), state, brake)
    assert int(state.alive.sum()) == env.num_arrivals  # one car per arrival point
    _, state, _, _, _ = env.step_env(jax.random.PRNGKey(1), state, brake)
    _, state, _, _, _ = env.step_env(jax.random.PRNGKey(2), state, brake)
    assert int(state.alive.sum()) == env.num_agents  # never more than num_agents


def _arrival_stats(arrived, route, num_arrivals, routes_per_arrival):
    """Per arrival point, the arrival rate and the share of each of its routes."""
    rates, route_shares = [], []
    for a in range(num_arrivals):
        here = arrived & (route // routes_per_arrival == a)
        rates.append(here.any(-1).mean())
        counts = np.bincount(
            route[here] % routes_per_arrival, minlength=routes_per_arrival
        )
        route_shares.append(counts / counts.sum())
    return np.array(rates), np.array(route_shares)


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_arrivals_match_reference_distribution(difficulty):
    # The transition test copies the port's arriving cars into the reference, so
    # this compares the arrival process itself, from an empty road.
    env = TrafficJunctionGrid(difficulty=difficulty, add_rate=0.5)
    samples = 4000
    _, state = env.reset(jax.random.PRNGKey(0))
    added = jax.vmap(env._add_cars, in_axes=(0, None))(
        jax.random.split(jax.random.PRNGKey(1), samples), state
    )
    port_arrived = np.asarray(added.alive)
    port_route = np.asarray(added.route)

    ref = _reference(env)
    np.random.seed(0)
    ref_arrived = np.zeros((samples, env.num_agents), dtype=bool)
    ref_route = np.zeros((samples, env.num_agents), dtype=int)
    for i in range(samples):
        ref.alive_mask[:] = 0
        ref.cars_in_sys = 0
        ref._add_cars()
        ref_arrived[i] = ref.alive_mask == 1
        ref_route[i] = ref.route_id

    rpa = env.routes_per_arrival
    port_rates, port_shares = _arrival_stats(
        port_arrived, port_route, env.num_arrivals, rpa
    )
    ref_rates, ref_shares = _arrival_stats(
        ref_arrived, ref_route, env.num_arrivals, rpa
    )
    # Binomial standard errors are about 0.008 for the rates and at most 0.03
    # for the route shares.
    np.testing.assert_allclose(port_rates, 0.5, atol=0.04)
    np.testing.assert_allclose(ref_rates, 0.5, atol=0.04)
    np.testing.assert_allclose(port_shares, 1 / rpa, atol=0.07)
    np.testing.assert_allclose(ref_shares, 1 / rpa, atol=0.07)
    # The car taking each arrival is a random one off the road, in both.
    for arrived in (port_arrived, ref_arrived):
        slot_share = arrived.sum(0) / arrived.sum()
        np.testing.assert_allclose(slot_share, 1 / env.num_agents, atol=0.02)


def test_first_arrival_points_take_the_last_free_slot():
    # The reference adds cars arrival point by arrival point and stops once
    # num_agents cars are on the road.
    env = TrafficJunctionGrid(difficulty="medium", add_rate=1.0)
    _, state = env.reset(jax.random.PRNGKey(0))
    alive = jnp.ones(env.num_agents, dtype=bool).at[3].set(False)
    state = state.replace(alive=alive)
    for k in range(20):
        added = env._add_cars(jax.random.PRNGKey(k), state)
        assert bool(added.alive.all())
        assert int(added.route[3]) // env.routes_per_arrival == 0


def test_collision_penalty_and_success():
    env = TrafficJunctionGrid(difficulty="easy", add_rate=0.0)
    _, state = env.reset(jax.random.PRNGKey(0))
    # Car 0 is one cell before car 1's cell on the same route; car 0 drives in.
    route = env.routes[0]
    state = state.replace(
        alive=state.alive.at[:2].set(True),
        route=state.route.at[:2].set(0),
        route_pos=state.route_pos.at[:2].set(jnp.array([1, 2])),
        car_loc=state.car_loc.at[:2].set(route[jnp.array([1, 2])]),
        wait=state.wait.at[:2].set(jnp.array([3, 1])),
    )
    actions = {a: jnp.int32(BRAKE) for a in env.agents}
    actions["agent_0"] = jnp.int32(GAS)
    _, state, reward, _, info = env.step_env(jax.random.PRNGKey(0), state, actions)
    np.testing.assert_allclose(float(reward["agent_0"]), -10 - 0.01 * 4, rtol=1e-6)
    np.testing.assert_allclose(float(reward["agent_1"]), -10 - 0.01 * 2, rtol=1e-6)
    assert all(float(reward[a]) == 0.0 for a in env.agents[2:])
    assert not bool(info["success"].any())

    # Success stays false for the rest of the episode.
    actions["agent_0"] = jnp.int32(BRAKE)
    state = state.replace(car_loc=state.car_loc.at[0].set(route[3]))
    _, state, _, _, info = env.step_env(jax.random.PRNGKey(1), state, actions)
    assert not bool(info["success"].any())


def test_car_leaves_at_end_of_route():
    env = TrafficJunctionGrid(difficulty="easy", add_rate=0.0)
    _, state = env.reset(jax.random.PRNGKey(0))
    end = int(env.route_lengths[1]) - 1
    state = state.replace(
        alive=state.alive.at[0].set(True),
        route=state.route.at[0].set(1),
        route_pos=state.route_pos.at[0].set(end),
        car_loc=state.car_loc.at[0].set(env.routes[1, end]),
        wait=state.wait.at[0].set(5),
    )
    actions = {a: jnp.int32(GAS) for a in env.agents}
    obs, state, reward, _, info = env.step_env(jax.random.PRNGKey(0), state, actions)
    assert not bool(state.alive[0])
    assert bool(info["completed"][0]) and not bool(info["completed"][1:].any())
    assert not bool(info["alive"].any())
    np.testing.assert_array_equal(state.car_loc[0], [0, 0])
    assert int(state.wait[0]) == 0
    assert float(reward["agent_0"]) == 0.0
    assert not bool(obs["agent_0"].any())


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_difficulty_defaults_and_overrides(difficulty):
    env = make("traffic_junction_grid", difficulty=difficulty)
    for name, value in DIFFICULTIES[difficulty].items():
        assert getattr(env, name) == value
    env = make(
        "traffic_junction_grid", difficulty=difficulty, num_agents=4, max_steps=11
    )
    assert env.num_agents == 4 and env.max_steps == 11


def test_invalid_arguments():
    with pytest.raises(ValueError):
        TrafficJunctionGrid(difficulty="expert")
    with pytest.raises(ValueError):
        TrafficJunctionGrid(difficulty="medium", dim=7)
    with pytest.raises(ValueError):
        TrafficJunctionGrid(difficulty="hard", dim=16)


def test_jit_vmap_and_auto_reset():
    env = make("traffic_junction_grid", difficulty="medium")
    batch = 64
    obs, state = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(0), batch))

    def _step(carry, key):
        obs, state = carry
        key_act, key_step = jax.random.split(key)
        keys = jax.random.split(key_act, env.num_agents)
        actions = {
            a: jax.random.randint(keys[i], (batch,), 0, 2)
            for i, a in enumerate(env.agents)
        }
        obs, state, reward, done, info = jax.vmap(env.step)(
            jax.random.split(key_step, batch), state, actions
        )
        return (obs, state), done["__all__"]

    (obs, state), dones = jax.lax.scan(
        jax.jit(_step), (obs, state), jax.random.split(jax.random.PRNGKey(1), 100)
    )
    # Episodes of 40 steps: done after steps 40 and 80 of 100.
    assert int(dones.sum()) == 2 * batch
    for a in env.agents:
        assert obs[a].shape == (batch, *env.observation_space(a).shape)


def test_accepts_actions_with_a_trailing_dimension():
    """The IPPO and MAPPO baselines pass each action as a (1,) array."""
    env = make("traffic_junction_grid", difficulty="medium")
    _, state = env.reset(jax.random.PRNGKey(0))
    scalar = {a: jnp.int32(i % 2) for i, a in enumerate(env.agents)}
    vector = {a: v[None] for a, v in scalar.items()}
    obs_s, state_s, reward_s, _, _ = env.step_env(jax.random.PRNGKey(1), state, scalar)
    obs_v, state_v, reward_v, _, _ = env.step_env(jax.random.PRNGKey(1), state, vector)
    for a in env.agents:
        np.testing.assert_array_equal(obs_s[a], obs_v[a])
        assert float(reward_s[a]) == float(reward_v[a])
    jax.tree.map(np.testing.assert_array_equal, state_s, state_v)

"""Steps per second of the JAX grid Traffic Junction against the IC3Net reference.

Run from this directory: ``python benchmark.py``.
"""

import time

import jax
import numpy as np
from test_traffic_junction_grid import _reference

from jaxmarl.environments.traffic_junction_grid import TrafficJunctionGrid


def jax_steps_per_second(env, num_envs, num_steps=1000):
    def _rollout(key):
        key_reset, key_steps = jax.random.split(key)
        _, state = jax.vmap(env.reset)(jax.random.split(key_reset, num_envs))

        def _step(state, key):
            key_act, key_step = jax.random.split(key)
            acts = jax.random.randint(key_act, (env.num_agents, num_envs), 0, 2)
            actions = {a: acts[i] for i, a in enumerate(env.agents)}
            _, state, reward, _, _ = jax.vmap(env.step)(
                jax.random.split(key_step, num_envs), state, actions
            )
            return state, reward[env.agents[0]]

        _, rewards = jax.lax.scan(_step, state, jax.random.split(key_steps, num_steps))
        return rewards

    rollout = jax.jit(_rollout)
    rollout(jax.random.PRNGKey(0)).block_until_ready()
    start = time.perf_counter()
    rollout(jax.random.PRNGKey(1)).block_until_ready()
    return num_envs * num_steps / (time.perf_counter() - start)


def reference_steps_per_second(env, num_steps=20_000):
    ref = _reference(env)
    rng = np.random.default_rng(0)
    t = 0
    start = time.perf_counter()
    for _ in range(num_steps):
        ref.step(rng.integers(2, size=env.num_agents))
        t += 1
        if t == env.max_steps:
            ref.reset()
            t = 0
    return num_steps / (time.perf_counter() - start)


if __name__ == "__main__":
    print(f"JAX devices: {jax.devices()}")
    for difficulty in ("easy", "medium", "hard"):
        env = TrafficJunctionGrid(difficulty=difficulty)
        ref_sps = reference_steps_per_second(env)
        print(f"{difficulty}: reference {ref_sps:,.0f} steps/s")
        for num_envs in (1, 128, 1024, 8192):
            sps = jax_steps_per_second(env, num_envs)
            print(f"  JAX, {num_envs:>5} envs: {sps:,.0f} steps/s")

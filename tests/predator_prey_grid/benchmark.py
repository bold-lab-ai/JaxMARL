"""Steps per second of the JAX grid Predator-Prey against the IC3Net reference.

Run from this directory: ``python benchmark.py``.
"""

import time

import jax
import numpy as np
from ic3net_predator_prey import PredatorPreyEnv
from test_predator_prey_grid import CONFIGS, _reference

from jaxmarl.environments.predator_prey_grid import PredatorPreyGrid


def jax_steps_per_second(env, num_envs, num_steps=1000):
    def _rollout(key):
        key_reset, key_steps = jax.random.split(key)
        _, state = jax.vmap(env.reset)(jax.random.split(key_reset, num_envs))

        def _step(state, key):
            key_act, key_step = jax.random.split(key)
            acts = jax.random.randint(
                key_act, (env.num_agents, num_envs), 0, env.num_actions
            )
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
    ref: PredatorPreyEnv = _reference(env)
    rng = np.random.default_rng(0)
    ref.reset()
    t = 0
    start = time.perf_counter()
    for _ in range(num_steps):
        _, _, over, _ = ref.step(rng.integers(env.num_actions, size=env.num_agents))
        t += 1
        if over or t == env.max_steps:
            ref.reset()
            t = 0
    return num_steps / (time.perf_counter() - start)


if __name__ == "__main__":
    print(f"JAX devices: {jax.devices()}")
    for name, config in zip(("easy", "medium", "hard"), CONFIGS):
        env = PredatorPreyGrid(mode="cooperative", **config)
        ref_sps = reference_steps_per_second(env)
        print(f"{name}: reference {ref_sps:,.0f} steps/s")
        for num_envs in (1, 128, 1024, 8192):
            sps = jax_steps_per_second(env, num_envs)
            print(f"  JAX, {num_envs:>5} envs: {sps:,.0f} steps/s")

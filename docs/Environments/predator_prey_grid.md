# Grid Predator-Prey

A JAX implementation of the grid Predator-Prey task from [Learning when to Communicate at Scale in Multiagent Cooperative and Competitive Tasks](https://arxiv.org/abs/1812.09755) (IC3Net, ICLR 2019). Along with Traffic Junction, it is a standard benchmark for learned communication, used by CommNet, IC3Net, TarMAC and later methods. It ports the [reference implementation](https://github.com/IC3Net/IC3Net/tree/master/ic3net-envs), and its transitions are tested against that code.

`num_agents` predators search a `dim` x `dim` grid for a single fixed prey. Each predator sees only a small square window around itself, so finding the prey quickly rewards sharing what each predator has seen.

This is a different task from `JaxRobotarium_predator_prey`, which is continuous-space.

## Observation, action space and reward

Each predator observes its `(2 * vision + 1) x (2 * vision + 1)` window, flattened to a vector of length `(2 * vision + 1)^2 * vocab_size`: for every cell, a one-hot id of that grid cell (or *outside* for cells beyond the edge), plus the number of predators and prey in it. `vocab_size = dim * dim + 4`.

| Action | Meaning |
| --- | --- |
| 0 | Up |
| 1 | Right |
| 2 | Down |
| 3 | Left |
| 4 | Stay (absent with `stay=False`) |

Predators can't leave the grid, and stop moving once they reach the prey. Every predator not on the prey receives `-0.05` per step. A predator on the prey receives, with `n` predators on the prey:

| `mode` | Reward on the prey | Episode ends |
| --- | --- | --- |
| `cooperative` | `0.05 * n` | after `max_steps` |
| `competitive` | `0.05 / n` | after `max_steps` |
| `mixed` (default) | `0` | when all predators have reached the prey, or after `max_steps` |

`info["success"]` is true on steps where every predator is on the prey.

## Configurations

The defaults are the paper's easy setting.

| Paper setting | Arguments |
| --- | --- |
| Easy | `num_agents=3, dim=5, vision=0, max_steps=20` |
| Medium | `num_agents=5, dim=10, vision=1, max_steps=40` |
| Hard | `num_agents=10, dim=20, vision=1, max_steps=80` |

## Usage

```python
import jax
from jaxmarl import make

key = jax.random.PRNGKey(0)
env = make("predator_prey_grid", mode="cooperative", num_agents=5, dim=10, vision=1, max_steps=40)

key, key_reset = jax.random.split(key)
obs, state = env.reset(key_reset)

for _ in range(10):
    key, key_act, key_step = jax.random.split(key, 3)
    key_act = jax.random.split(key_act, env.num_agents)
    actions = {
        agent: env.action_space(agent).sample(key_act[i])
        for i, agent in enumerate(env.agents)
    }
    obs, state, reward, done, info = env.step(key_step, state, actions)
```

`init_render` and `update_render` draw the grid with matplotlib, for use with `jaxmarl.viz.visualizer.Visualizer`.

## Performance

Environment steps per second (one step of all predators) for random actions, with the cooperative reward, measured by `tests/predator_prey_grid/benchmark.py` on an NVIDIA L40S. The reference runs one environment in Python on the same node's Intel Xeon Gold 6448Y.

| Setting | Reference | JAX, 1 env | JAX, 128 envs | JAX, 1,024 envs | JAX, 8,192 envs |
| --- | --- | --- | --- | --- | --- |
| Easy (5x5, 3 predators) | 25,700 | 65,800 | 9.2M | 59M | 263M |
| Medium (10x10, 5 predators) | 20,000 | 61,600 | 8.1M | 50M | 180M |
| Hard (20x20, 10 predators) | 4,800 | 55,000 | 6.1M | 30M | 55M |

## Baseline results

IPPO and MAPPO (`baselines/IPPO/ippo_rnn.py`, `baselines/MAPPO/mappo_rnn.py`) on the paper's medium setting in mixed mode, 20 seeds of 2e7 environment steps each, with the configs `ippo_rnn_predator_prey_grid.yaml` and `mappo_homogenous_rnn_predator_prey_grid.yaml`. Values are the mean ± standard deviation over seeds of the final greedy test episodes.

| Algorithm | Return | Success rate | Episode length | Wall time (L40S, 20 seeds) |
| --- | --- | --- | --- | --- |
| IPPO | -0.79 ± 0.03 | 0.92 ± 0.03 | 26.2 ± 0.8 | 21 min |
| MAPPO | -0.80 ± 0.03 | 0.80 ± 0.04 | 26.1 ± 1.4 | 93 min |

The IC3Net paper reports the number of steps to reach the prey: 28.0 for independent agents without communication (IC) and 13.1 for CommNet. IPPO plateaus early at about the IC level; MAPPO is still improving at 2e7 steps. MAPPO's 20 seeds don't fit in an L40S's 48 GB at once, so they were run 5 at a time.

```bash
python baselines/IPPO/ippo_rnn.py --config-name ippo_rnn_predator_prey_grid NUM_SEEDS=20
python baselines/MAPPO/mappo_rnn.py --config-name mappo_homogenous_rnn_predator_prey_grid NUM_SEEDS=5
```

## Correspondence with the reference implementation

`tests/predator_prey_grid/test_predator_prey_grid.py` steps this environment and the reference side by side, for each mode and the three paper settings, and checks that observations, rewards, `reached_prey`, termination and success agree at every step. Differences from the reference:

- The time limit is part of the environment; the reference leaves it to the training loop.
- Observations are flat vectors, like those of the other JaxMARL environments, so they work with the standard IPPO and MAPPO baselines. They are the reference's `(2v+1, 2v+1, vocab)` observations flattened in that order. The reference also declares its space as `(vocab, 2v+1, 2v+1)` with `high=1`, but returns `(2v+1, 2v+1, vocab)` arrays with counts up to `num_agents`.
- The reference's moving prey and prey agents (`--moving_prey`, `--enemy_comm`) are not implemented. The reference itself raises `NotImplementedError` for a moving prey.
- The paper's text gives a mixed-mode reward of `0.05` on the prey; the reference code, which produced the paper's results, gives `0`. This follows the code.

## Citation

If you use this environment, please cite:

    @inproceedings{singh2019learning,
        title={Learning when to Communicate at Scale in Multiagent Cooperative and Competitive Tasks},
        author={Singh, Amanpreet and Jain, Tushar and Sukhbaatar, Sainbayar},
        booktitle={International Conference on Learning Representations},
        year={2019}
    }

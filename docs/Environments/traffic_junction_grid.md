# Grid Traffic Junction

A JAX implementation of the Traffic Junction task from [Learning Multiagent Communication with Backpropagation](https://arxiv.org/abs/1605.07736) (CommNet, NeurIPS 2016), in the three versions of [Learning when to Communicate at Scale in Multiagent Cooperative and Competitive Tasks](https://arxiv.org/abs/1812.09755) (IC3Net, ICLR 2019). Along with grid Predator-Prey, it is a standard benchmark for learned communication, used by CommNet, IC3Net, TarMAC and later methods. It ports the IC3Net [reference implementation](https://github.com/IC3Net/IC3Net/tree/master/ic3net-envs), and its transitions are tested against that code.

Cars enter at the ends of the roads, follow a fixed route through one or more junctions, and leave at its end. Each car sees little or nothing beyond its own cell, so avoiding collisions at the junctions rewards coordination.

## Observation, action space and reward

Each car observes a flat vector:

| Entries | Value |
| --- | --- |
| 1 | Its last action, `0` (GAS) or `1` (BRAKE) |
| 1 | Its route id, normalised to `[0, 1]` |
| `(2 * vision + 1)^2 * vocab_size` | For every cell of its window, a one-hot id of that road cell (or *outside*), plus the number of cars in it |

Cars that are off the road observe zeros.

| Action | Meaning |
| --- | --- |
| 0 | GAS: advance one cell along the route |
| 1 | BRAKE: stay |

A car on the road receives `-0.01 * wait` per step, where `wait` is the number of steps it has spent on the road, and `-10` for each step in which it shares a cell with another car. Cars off the road receive 0. Episodes end after `max_steps`. `info["success"]` is true when no collision has happened in the episode. As in the reference, `info["alive"]` marks the cars on the road after the step, and `info["completed"]` the cars that left at the end of their route during it; methods that mask cars off the road, like the IC3Net baselines, use them.

At each step, a car enters at each arrival point with probability `add_rate`, while fewer than `num_agents` cars are on the road; entering cars take a random route from that arrival point. The agents are a fixed pool of `num_agents` car slots, which are reused as cars leave and enter.

## Configurations

`difficulty` selects the paper's layout and defaults for the other arguments (Table 4 of the IC3Net paper and the reference's training commands); any of them can be overridden.

| `difficulty` | Layout | Grid | Routes | `num_agents` | `max_steps` | `add_rate` |
| --- | --- | --- | --- | --- | --- | --- |
| `easy` (default) | 1 junction, two one-way roads | 7x7 (`dim=6`) | 2 | 5 | 20 | 0.3 |
| `medium` | 1 junction, two-way roads | 14x14 | 12 | 10 | 40 | 0.2 |
| `hard` | 4 junctions, two-way roads | 18x18 | 56 | 20 | 80 | 0.05 |

`vision` defaults to 0, as in the paper. The paper trains with a curriculum that raises `add_rate` from 0.1, 0.05 and 0.02 respectively to the values above; the defaults are the final values.

## Usage

```python
import jax
from jaxmarl import make

key = jax.random.PRNGKey(0)
env = make("traffic_junction_grid", difficulty="medium")

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

`init_render` and `update_render` draw the roads and cars with matplotlib, for use with `jaxmarl.viz.visualizer.Visualizer`.

## Performance

Environment steps per second (one step of all cars) for random actions, measured by `tests/traffic_junction_grid/benchmark.py` on an NVIDIA L40S. The reference runs one environment in Python on the same node's Intel Xeon Gold 6448Y.

| `difficulty` | Reference | JAX, 1 env | JAX, 128 envs | JAX, 1,024 envs | JAX, 8,192 envs |
| --- | --- | --- | --- | --- | --- |
| `easy` | 10,800 | 23,800 | 2.7M | 21M | 169M |
| `medium` | 6,400 | 15,700 | 1.6M | 12M | 97M |
| `hard` | 3,300 | 9,200 | 0.9M | 6.8M | 47M |

## Baseline results

IPPO and MAPPO (`baselines/IPPO/ippo_rnn.py`, `baselines/MAPPO/mappo_rnn.py`) on `difficulty="medium"`, 20 seeds of 2e7 environment steps each, with the configs `ippo_rnn_traffic_junction_grid.yaml` and `mappo_homogenous_rnn_traffic_junction_grid.yaml`. Values are the mean ± standard deviation over seeds of the final greedy test episodes.

| Algorithm | Return | Success rate | Wall time (L40S, 20 seeds) |
| --- | --- | --- | --- |
| IPPO | -21.2 ± 1.1 | 0.03 ± 0.01 | 24 min |
| MAPPO | -10.2 ± 0.5 | 0.05 ± 0.01 | 51 min |

Both learn to keep cars moving, which removes most of the waiting penalty, but neither avoids collisions. This matches the IC3Net paper, where agents without communication (IRIC) reach 3.4% success on the medium version, against 89.3% for IC3Net.

```bash
python baselines/IPPO/ippo_rnn.py --config-name ippo_rnn_traffic_junction_grid NUM_SEEDS=20
python baselines/MAPPO/mappo_rnn.py --config-name mappo_homogenous_rnn_traffic_junction_grid NUM_SEEDS=20
```

## Correspondence with the reference implementation

`tests/traffic_junction_grid/test_traffic_junction_grid.py` checks that the routes of all three versions are identical to the reference's. Car arrivals are random, so the test replays each step of this environment through the reference, with the same state, actions and arriving cars, and checks that car positions, waiting times, rewards, collisions and observations agree. Differences from the reference:

- The reference sizes the easy version's road ids from `dim` rather than the enlarged `dim + 1` grid, so two road cells share the *outside* and road ids. Here every road cell has its own id. The test applies the same change to the reference.
- Cars that are off the road are not counted in observations. The reference counts them at cell `(0, 0)`, which is never road and only falls inside a window for large `vision`.
- The paper's text gives 60 steps for the hard version; the reference's training command, which produced its results, uses 80. This follows the command.
- `add_rate` is fixed for an environment; the reference's curriculum is left to the training loop.

## Citation

If you use this environment, please cite:

    @inproceedings{sukhbaatar2016learning,
        title={Learning Multiagent Communication with Backpropagation},
        author={Sukhbaatar, Sainbayar and Szlam, Arthur and Fergus, Rob},
        booktitle={Advances in Neural Information Processing Systems},
        year={2016}
    }

    @inproceedings{singh2019learning,
        title={Learning when to Communicate at Scale in Multiagent Cooperative and Competitive Tasks},
        author={Singh, Amanpreet and Jain, Tushar and Sukhbaatar, Sainbayar},
        booktitle={International Conference on Learning Representations},
        year={2019}
    }

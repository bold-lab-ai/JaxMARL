# Grid Traffic Junction

A JAX implementation of the Traffic Junction task from [Learning Multiagent Communication with Backpropagation](https://arxiv.org/abs/1605.07736) (CommNet, NeurIPS 2016), in the three versions of [Learning when to Communicate at Scale in Multiagent Cooperative and Competitive Tasks](https://arxiv.org/abs/1812.09755) (IC3Net, ICLR 2019). It is a standard benchmark for learned communication, used by CommNet, IC3Net, TarMAC and later methods. It ports the IC3Net [reference implementation](https://github.com/IC3Net/IC3Net/tree/master/ic3net-envs), and its transitions are tested against that code.

Cars enter at the ends of the roads, follow a fixed route through one or more junctions, and leave at its end. Each car sees little or nothing beyond its own cell, so avoiding collisions at the junctions rewards coordination.

<p align="center">
  <img src="https://github.com/FLAIROx/JaxMARL/blob/main/docs/imgs/traffic_junction_grid.gif?raw=true" alt="Traffic Junction" width="300">
</p>

A successful episode of trained agents on the medium version.

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

A car on the road receives `-0.01 * wait` per step, where `wait` is the number of steps it has spent on the road, and `-10` for each step in which it shares a cell with another car. Cars off the road receive 0. Episodes end after `max_steps`. `info["success"]` is true when no collision has happened in the episode. As in the reference, `info["alive"]` marks the cars on the road after the step, including cars that entered during it, and `info["completed"]` the cars that left at the end of their route during it. A trainer can use them to mask cars off the road, as the reference's trainer does.

At each step, a car enters at each arrival point with probability `add_rate`, while fewer than `num_agents` cars are on the road; entering cars take a random route from that arrival point. The agents are a fixed pool of `num_agents` car slots, which are reused as cars leave and enter.

## Configurations

`difficulty` selects the paper's layout and defaults for the other arguments, from Table 4 of the IC3Net paper and, for `max_steps`, its text for `easy` and `medium` and the reference's command for `hard`; any of them can be overridden.

| `difficulty` | Layout | Grid | Routes | `num_agents` | `max_steps` | `add_rate` |
| --- | --- | --- | --- | --- | --- | --- |
| `easy` (default) | 1 junction, two one-way roads | 7x7 (`dim=6`) | 2 | 5 | 20 | 0.3 |
| `medium` | 1 junction, two-way roads | 14x14 | 12 | 10 | 40 | 0.2 |
| `hard` | 4 junctions, two-way roads | 18x18 | 56 | 20 | 80 | 0.05 |

`vision` defaults to 0, as in the paper. The paper trains with a curriculum that raises `add_rate` from 0.1, 0.05 and 0.02 respectively to the values above; the defaults are the final values. For `hard`, the paper's text gives a `max_steps` of 60 and the reference's command 80; the default follows the command.

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

## Visualisation

`init_render` and `update_render` draw the environment with matplotlib, for use with `Visualizer`:

```python
import jax
from jaxmarl import make
from jaxmarl.viz.visualizer import Visualizer

key = jax.random.PRNGKey(0)
env = make("traffic_junction_grid", difficulty="medium")

key, key_reset = jax.random.split(key)
obs, state = env.reset(key_reset)
state_seq = [state]
for _ in range(40):
    key, key_act, key_step = jax.random.split(key, 3)
    key_act = jax.random.split(key_act, env.num_agents)
    actions = {
        agent: env.action_space(agent).sample(key_act[i])
        for i, agent in enumerate(env.agents)
    }
    obs, state, reward, done, info = env.step(key_step, state, actions)
    state_seq.append(state)

viz = Visualizer(env, state_seq)
viz.animate(save_fname="traffic_junction.gif", view=False)
```

## Performance

Environment steps per second (one step of all cars, including every car's observation) for random actions, measured by `tests/traffic_junction_grid/benchmark.py` on an NVIDIA L40S as the median of three timed rollouts of 1,000 steps after compilation. The reference runs one environment in Python on the same node's Intel Xeon Gold 6448Y.

| `difficulty` | Reference | JAX, 1 env | JAX, 128 envs | JAX, 1,024 envs | JAX, 8,192 envs |
| --- | --- | --- | --- | --- | --- |
| `easy` | 10,800 | 20,700 | 2.3M | 15M | 106M |
| `medium` | 6,500 | 14,100 | 1.3M | 7.8M | 11M |
| `hard` | 3,300 | 8,500 | 0.63M | 2.1M | 3.1M |

## Baselines

IPPO and MAPPO configurations for the medium version are in [`ippo_rnn_traffic_junction_grid.yaml`](https://github.com/FLAIROx/JaxMARL/tree/main/baselines/IPPO/config/ippo_rnn_traffic_junction_grid.yaml) and [`mappo_homogenous_rnn_traffic_junction_grid.yaml`](https://github.com/FLAIROx/JaxMARL/tree/main/baselines/MAPPO/config/mappo_homogenous_rnn_traffic_junction_grid.yaml):

```bash
python baselines/IPPO/ippo_rnn.py --config-name ippo_rnn_traffic_junction_grid
python baselines/MAPPO/mappo_rnn.py --config-name mappo_homogenous_rnn_traffic_junction_grid
```

They train at the fixed default `add_rate`, without the paper's curriculum, and unlike the reference's trainer don't mask the steps a car slot spends off the road.

## Correspondence with the reference implementation

`tests/traffic_junction_grid/test_traffic_junction_grid.py` checks that the routes of all three versions are identical to the reference's. Car arrivals are random, so the test replays each step of this environment through the reference, with the same state, actions and arriving cars, and checks that car positions, waiting times, rewards, collisions and observations agree. Differences from the reference:

- Every road cell has its own id; in the reference's easy version, one road cell has the *outside* id.
- Cars that are off the road are not counted in observations.
- The paper's reward scales the collision penalty with the number of collisions; the reference code, followed here, gives `-10` once per step however many cars share the cell.
- `add_rate` is fixed for an environment; the reference's curriculum is left to the training loop.
- The time limit is part of the environment; the reference never ends an episode and leaves the limit to the training loop.
- Observations are flat vectors, like those of the other JaxMARL environments: the reference's `(last action, route id, window)` tuple, with the `(2v+1, 2v+1, vocab)` window flattened in that order.

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

# Learned Communication

Pure JAX implementations of:

* CommNet ([Sukhbaatar et al., 2016](https://arxiv.org/abs/1605.07736))
* IC3Net ([Singh et al., 2019](https://arxiv.org/abs/1812.09755))
* IC and IRIC, the no-communication controls from the IC3Net paper

`ic3net.py` follows the IC3Net [reference implementation](https://github.com/IC3Net/IC3Net), which trains all four with REINFORCE and a learned value baseline.

## 📡 Communication and non-communication

- **Algorithms:** CommNet and IC3Net agents communicate through a differentiable channel in their network, not through message actions in the environment: before each LSTM step, every agent adds the mean of the other agents' hidden states to its input. IC3Net agents also learn, at every step, whether to communicate at the next one. IC and IRIC are the same recurrent agents without the channel, trained on the team's mean return (IC) or on their own (IRIC).
- **Environments:** communication pays off only when agents can't coordinate otherwise. Traffic Junction cars can't see each other; on easy, success rises from 28% without communication to 85–89% with it. In Predator-Prey, predators that share where the prey is nearly halve the episode length. On the MPE tasks, communication doesn't improve on IC with these settings.

Supported environments:

- Traffic Junction (`traffic_junction_grid`): easy, medium and hard
- Predator-Prey (`predator_prey_grid`): easy, medium and hard
- MPE Simple Reference, Speaker-Listener (with `PAD_SPACES`) and Simple Spread with continuous actions

All agents share parameters, so other environments need agents with the same spaces, or `PAD_SPACES=True`, and actions that are all discrete or all continuous.

## ⚙️ Implementation Details

- Each update trains on whole episodes, with Monte Carlo returns and RMSprop, as in the reference.
- Continuous actions use a Gaussian policy with a learned log standard deviation.
- Unlike the reference, every update runs a fixed number of episodes (`NUM_ENVS`) rather than a minimum number of steps, and there is no curriculum, as in JaxMARL's other baselines.

## 🚀 Usage

If you have cloned JaxMARL and are in the repository root, select the method with `BASELINE` and the environment with `+alg`:

```bash
python baselines/learned-communication/ic3net.py +alg=ic3net_traffic_junction_medium BASELINE=ic3net NUM_SEEDS=20
```

The default configuration, including the `wandb` details (`wandb` is disabled by default), is in [`config.yaml`](https://github.com/FLAIROx/JaxMARL/tree/main/baselines/learned-communication/config/config.yaml), and the environment configurations in [`config/alg`](https://github.com/FLAIROx/JaxMARL/tree/main/baselines/learned-communication/config/alg). The main keys:

| Key | Meaning |
| --- | --- |
| `BASELINE` | `ic`, `iric`, `commnet` or `ic3net` |
| `MEAN_RATIO` | Weight of the team's mean return in each agent's return: 1 for `ic` and `commnet`, 0 for `iric` and `ic3net`, as in the reference |
| `COMM_ACTION_ONE` | IC3Net agents always communicate; the reference sets it for Traffic Junction |
| `PAD_SPACES` | Pads observations and action spaces to the largest, for agents with different spaces |
| `SEED_CHUNK` | Trains the `NUM_SEEDS` seeds this many at a time, to fit in memory |

## Citation

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

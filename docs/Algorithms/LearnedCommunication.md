# Learned-Communication Baselines

Pure JAX implementations of methods whose agents learn to communicate through a differentiable channel in their architecture, as opposed to environments with message actions:

* CommNet ([Sukhbaatar et al., 2016](https://arxiv.org/abs/1605.07736)): before each LSTM step, every agent adds the mean of the other agents' hidden states to its input.
* IC3Net ([Singh et al., 2019](https://arxiv.org/abs/1812.09755)): CommNet where each agent also chooses, at every step, whether to communicate at the next one.
* IC and IRIC: the no-communication controls from the IC3Net paper, independent recurrent agents trained on the team's mean return (IC) or on their own (IRIC).

`ic3net.py` ports the IC3Net [reference implementation](https://github.com/IC3Net/IC3Net), which trains all four with REINFORCE and a learned value baseline.

## 🚀 Usage

`BASELINE` selects the method and `+alg` the environment configuration:

```bash
python baselines/learned-communication/ic3net.py +alg=ic3net_traffic_junction_medium BASELINE=ic3net
python baselines/learned-communication/ic3net.py +alg=ic3net_predator_prey_medium BASELINE=commnet NUM_SEEDS=20
python baselines/learned-communication/ic3net.py +alg=ic3net_predator_prey_easy BASELINE=iric alg.LR=0.002
```

The configurations are the IC3Net paper's easy, medium and hard settings of `traffic_junction_grid` and `predator_prey_grid`, with the reference's training settings. Any environment with homogeneous agents and discrete actions works: all agents share parameters.

| Key | Meaning |
| --- | --- |
| `BASELINE` | `ic`, `iric`, `commnet` or `ic3net` |
| `MEAN_RATIO` | Weight of the team's mean return in each agent's return. By default 1 for `ic` and `commnet` and 0 for `iric` and `ic3net`, as in the reference |
| `NUM_ENVS`, `NUM_STEPS` | Each update runs `NUM_ENVS` episodes from reset for `NUM_STEPS` steps; steps after an episode ends are ignored |
| `COMM_PASSES` | Communication rounds per step |
| `COMM_ACTION_ONE` | IC3Net agents always communicate; the reference sets it for Traffic Junction |
| `DETACH_GAP` | Steps of backpropagation through time |
| `CURRICULUM` | Optional. Raises the environment argument `KEY` from `START` to `END` in steps of `STEP`, between the fractions `BEGIN` and `FINISH` of training. The Traffic Junction configurations raise `add_rate` as the reference does |

## ⚙️ Implementation details

The tests in `tests/baselines/test_ic3net.py` check the returns and one step of the policies against NumPy ports of the reference's loops. As in the reference:

- Each update trains on whole episodes, with Monte Carlo returns and RMSprop (`alpha=0.97`, `eps=1e-6`) without gradient clipping. Losses are summed over agents and averaged over environment steps.
- With `MEAN_RATIO` below 1, an agent's own return ends when its episode does: a car that leaves Traffic Junction does not inherit the rewards of the next car in its slot.
- Agents that are dead (`info["alive"]`, when the environment reports it) or silent neither send nor receive messages, and dead agents are masked out of the loss. An IC3Net agent's choice to communicate takes effect at the next step; no agent communicates at the first.
- Traffic Junction trains with the reference's curriculum: `add_rate` starts at the paper's lower value and rises in steps of 0.01 between epochs 250 and 1,250 of 2,000. Without it, all four methods settle on never braking.
- Reported statistics come from the training episodes.

Differences from the reference:

- Layers use the orthogonal initialisation of the other JaxMARL baselines rather than PyTorch's defaults.
- Every update runs a fixed number of episodes; the reference runs 16 processes that each collect at least 500 steps of whole episodes. The configurations match the reference's batch size for episodes that last until the time limit. Traffic Junction episodes always do, but Predator-Prey episodes end once every predator reaches the prey, so as the agents improve, the batch shrinks to fewer steps than the reference's.

## 📈 Results

All four methods on the medium versions, with the configurations above: 20 seeds each, trained for 1.6e8 environment steps (20,000 updates, the reference's 2,000 epochs). Values are the mean ± standard deviation over seeds of the training episodes in the last 2% of updates. The paper's values are the mean over 5 runs.

| Method | Traffic Junction success (%) | Paper | Predator-Prey steps | Paper |
| --- | --- | --- | --- | --- |
| IC | 4.7 ± 6.0 | 3.4 ± 0.5 | 28.0 ± 0.4 | 28.0 ± 0.74 |
| IRIC | 3.4 ± 0.1 | 3.4 ± 0.5 | 28.3 ± 0.3 | 28.1 ± 0.2 |
| CommNet | 30.7 ± 27.6 | 54.3 ± 14.2 | 13.8 ± 1.4 | 13.1 ± 0.01 |
| IC3Net | 67.0 ± 26.8 | 89.3 ± 2.5 | 16.3 ± 1.5 | 13.0 ± 0.02 |

- Without communication, IC and IRIC match the paper on both tasks. On Traffic Junction, 39 of their 40 seeds end at 3.2–3.5% success.
- On Traffic Junction, the methods rank as in the paper, but the communicating ones vary far more across seeds. Eight IC3Net seeds exceed 85% success and three stay below 25%; IC3Net's mean success was still rising at the end of training. Eight CommNet seeds stay below 6%, four reach 11–26% and eight reach 46–74%.
- On Predator-Prey, CommNet matches the paper but IC3Net does not overtake it. Part of the gap comes from the smaller batches of short episodes: with `NUM_ENVS=600`, which brings the batch close to the reference's 8,000 steps, 4 seeds of CommNet take 13.1 ± 0.03 steps and IC3Net 15.1 ± 0.8. IC3Net's agents communicate at 70% of steps.
- For comparison, IPPO and MAPPO on the same environments (see the environments' READMEs, 2e7 steps, greedy test episodes) reach 3–5% success on Traffic Junction and take 26 steps on Predator-Prey.

Wall time for 20 seeds on one L40S, run 10 at a time, including compilation:

| Method | Traffic Junction | Predator-Prey | Environment steps per second |
| --- | --- | --- | --- |
| IC, IRIC | 18 min | 40 min | 3.0e6 (TJ), 1.3e6 (PP) |
| CommNet, IC3Net | 61–64 min | 62 min | 8.3e5–8.7e5 |

```bash
python baselines/learned-communication/ic3net.py +alg=ic3net_traffic_junction_medium BASELINE=ic3net NUM_SEEDS=10
python baselines/learned-communication/ic3net.py +alg=ic3net_predator_prey_medium BASELINE=ic3net NUM_SEEDS=10
```

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

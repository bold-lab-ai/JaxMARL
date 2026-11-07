# Learned-Communication Baselines

Pure JAX implementations of:

* CommNet ([Sukhbaatar et al., 2016](https://arxiv.org/abs/1605.07736))
* IC3Net ([Singh et al., 2019](https://arxiv.org/abs/1812.09755))
* IC and IRIC, the no-communication controls from the IC3Net paper

The agents learn to communicate through a differentiable channel in their network, rather than through message actions in the environment. `ic3net.py` follows the IC3Net [reference implementation](https://github.com/IC3Net/IC3Net), which trains all four with REINFORCE and a learned value baseline.

Supported environments:

- Traffic Junction (`traffic_junction_grid`): the paper's easy, medium and hard versions
- Predator-Prey (`predator_prey_grid`): the paper's easy, medium and hard settings
- MPE Simple Reference
- MPE Speaker-Listener, whose agents have different observations and actions, with `PAD_SPACES`
- MPE Simple Spread with continuous actions

All agents share parameters, so other environments need agents with the same observation and action spaces, or `PAD_SPACES=True`, and action spaces that are all discrete or all continuous. Their optional `alive`, `completed` and `success` info entries must hold one value per agent. Every agent communicates with every other, so the script doesn't separate competing teams.

## ⚙️ Implementation Details

- CommNet: before each LSTM step, every agent adds the mean of the other agents' hidden states to its input. CommNet agents always communicate.
- IC3Net: CommNet where each agent also chooses, at every step, whether to communicate at the next one. Agents that are silent or dead (`info["alive"]`, when the environment reports it) neither send nor receive messages.
- IC and IRIC: independent recurrent agents, trained on the team's mean return (IC) or on their own (IRIC).
- Each update trains on whole episodes, with Monte Carlo returns and RMSprop (`alpha=0.97`, `eps=1e-6`) without gradient clipping, as in the reference. Losses are summed over agents and averaged over environment steps, and each step's loss is masked with the agents alive after it.
- Continuous actions use a Gaussian policy with a learned log standard deviation, as in the reference.
- Reported statistics come from the training episodes.

Differences from the reference:

- Layers use the orthogonal initialisation of the other JaxMARL baselines, and the LSTM flax's defaults, rather than PyTorch's defaults. Computation is in float32; the reference uses float64.
- Every update runs a fixed number of episodes, `NUM_ENVS`; the reference collects a fixed minimum number of steps, so its batches don't shrink when episodes end early.
- The entropy term is masked like the rest of the loss; the reference sums it over all agents. It is unused with the default `ENT_COEF` of 0.
- There is no curriculum (see [Configurations](#configurations)).

## 🚀 Usage

If you have cloned JaxMARL and are in the repository root, you can run the script with `BASELINE` selecting the method and `+alg` the environment configuration:

```bash
# IC3Net on medium Traffic Junction
python baselines/learned-communication/ic3net.py +alg=ic3net_traffic_junction_medium BASELINE=ic3net
# CommNet on medium Predator-Prey, 20 vmapped seeds
python baselines/learned-communication/ic3net.py +alg=ic3net_predator_prey_medium BASELINE=commnet NUM_SEEDS=20
# IRIC on easy Predator-Prey with another learning rate
python baselines/learned-communication/ic3net.py +alg=ic3net_predator_prey_easy BASELINE=iric alg.LR=0.002
```

Take a look at [`config.yaml`](https://github.com/FLAIROx/JaxMARL/tree/main/baselines/learned-communication/config/config.yaml) for the default configuration, including the `wandb` details (`wandb` is disabled by default), and at [`config/alg`](https://github.com/FLAIROx/JaxMARL/tree/main/baselines/learned-communication/config/alg) for the environment configurations. The main keys:

| Key | Meaning |
| --- | --- |
| `BASELINE` | `ic`, `iric`, `commnet` or `ic3net` |
| `MEAN_RATIO` | Weight of the team's mean return in each agent's return. By default 1 for `ic` and `commnet` and 0 for `iric` and `ic3net`, as in the reference. See [Results](#-results) for CommNet with 0 |
| `NUM_ENVS`, `NUM_STEPS` | Each update runs `NUM_ENVS` episodes from reset for `NUM_STEPS` steps; steps after an episode ends are ignored |
| `COMM_PASSES` | Communication rounds per step |
| `COMM_ACTION_ONE` | IC3Net agents always communicate; the reference sets it for Traffic Junction |
| `DETACH_GAP` | Steps of backpropagation through time |
| `SEED_CHUNK` | Trains the `NUM_SEEDS` seeds this many at a time, to fit in memory, with the same seeds as all at once |
| `LSTM` | By default, IC and IRIC use an LSTM, as in the paper; `False` gives the tanh RNN of the reference's commands. CommNet and IC3Net always use an LSTM |
| `PAD_SPACES` | Pads every agent's observation to the largest, followed by a one-hot agent id, and every agent's action space to the largest, masking the actions or action dimensions it doesn't have. Needed when agents have different spaces |
| `ACTION_SCALE` | For continuous actions: samples are multiplied by it, clipped to [-1, 1] and mapped onto the action space's bounds, as in the reference |

### Configurations

The Traffic Junction and Predator-Prey configurations follow the IC3Net paper's training settings, and the reference's commands where the paper doesn't give one, except for the curriculum:

| Setting | Paper | Reference's commands | Configurations |
| --- | --- | --- | --- |
| Traffic Junction learning rate | 0.003 | 0.001 | 0.003 |
| Predator-Prey learning rate | Tuned, not given | 0.001 | 0.001 |
| Predator-Prey training | 1,000 epochs | 2,000 epochs | 1,000 epochs |
| IC and IRIC network | LSTM | tanh RNN | LSTM |
| Hard Traffic Junction `max_steps` | 60 | 80 | 60 |
| Traffic Junction `add_rate` | Raised from 0.1, 0.05 and 0.02 to 0.3, 0.2 and 0.05 | The same, but medium's curriculum is disabled | Fixed at 0.3, 0.2 and 0.05 |

No other JaxMARL baseline trains with a curriculum, so Traffic Junction trains at the environment's fixed `add_rate`, as IPPO and MAPPO do. The MPE configurations use the Predator-Prey settings and aren't tuned for MPE.

## 📈 Results

All four methods with the configurations above, 20 seeds each, trained for 20,000 updates of 8,000 simulated steps on Traffic Junction (the paper's 2,000 epochs) and 10,000 on Predator-Prey (the paper's 1,000) and MPE. Values are the mean ± sample standard deviation over seeds of the training episodes in the last 2% of updates, followed by the bootstrap 95% confidence interval of the mean. The paper's values are over 5 runs.

Traffic Junction success (%). The Paper columns give the paper's Table 1 values, except in the numbered cells: there the reference code doesn't reach the paper's value either, so they give the reference code's value instead (see [Comparison with the reference code](#comparison-with-the-reference-code)), and the note gives the paper's.

| Method | Easy | Paper | Medium | Paper | Hard | Paper |
| --- | --- | --- | --- | --- | --- | --- |
| IC | 28.2 ± 0.1 [28.1, 28.2] | 30.2 ± 0.4 | 3.3 ± 0.2 [3.2, 3.4] | 3.4 ± 0.5 | 14.1 ± 2.0 [13.2, 14.9] | 13.6 ± 2.2² |
| IRIC | 28.2 ± 0.1 [28.1, 28.2] | 29.8 ± 0.7 | 3.4 ± 0.1 [3.4, 3.4] | 3.4 ± 0.5 | 16.3 ± 0.2 [16.2, 16.4] | 16.6 ± 0.2² |
| CommNet | 89.4 ± 15.7 [81.8, 94.5] | 93.0 ± 4.2 | 6.4 ± 13.2 [3.3, 12.3] | 13.4 ± 17.3¹ | 9.0 ± 5.0 [6.8, 11.2] | 8.3 ± 7.3² |
| CommNet, `MEAN_RATIO=0`, medium and hard only | | | 94.9 ± 3.2 [93.4, 96.2] | | 71.8 ± 13.8 [64.8, 76.1] | |
| IC3Net | 85.5 ± 7.0 [82.5, 88.5] | 93.0 ± 3.7 | 90.6 ± 7.7 [87.1, 93.7] | 89.3 ± 2.5 | 58.1 ± 16.6 [50.6, 64.6] | 37.2 ± 21.6² |

1. The reference code with the paper's curriculum, trained for 1,000 epochs. The paper reports 54.3 ± 14.2.
2. The reference code at the same settings, trained for 1,000 epochs. The paper reports 47.0 ± 2.9, 35.0 ± 0.6, 50.2 ± 3.5 and 72.4 ± 9.6.

Predator-Prey episode length, capped at 20, 40 and 80 steps (lower is better):

| Method | Easy | Paper | Medium | Paper | Hard | Paper |
| --- | --- | --- | --- | --- | --- | --- |
| IC | 16.4 ± 0.4 [16.2, 16.5] | 16.4 ± 0.49 | 27.7 ± 0.7 [27.4, 28.0] | 28.0 ± 0.74 | 76.1 ± 1.7 [75.4, 76.8] | 77.4 ± 0.8 |
| IRIC | 16.4 ± 0.6 [16.1, 16.6] | 16.5 ± 0.1 | 28.2 ± 0.4 [28.0, 28.4] | 28.1 ± 0.2 | 75.3 ± 1.5 [74.7, 75.9] | 75.0 ± 1.4 |
| CommNet | 9.4 ± 0.1 [9.3, 9.4] | 9.1 ± 0.1 | 14.4 ± 1.5 [13.9, 15.2] | 13.1 ± 0.01 | 76.5 ± 1.9 [75.7, 77.3] | 76.5 ± 1.3 |
| IC3Net | 9.4 ± 0.1 [9.3, 9.5] | 8.9 ± 0.02 | 16.7 ± 2.1 [15.9, 17.7] | 13.0 ± 0.02 | 66.2 ± 4.9 [64.0, 68.2] | 52.4 ± 3.4 |

- On the easy versions, communication raises Traffic Junction success from 28% to 85–89% and nearly halves Predator-Prey episodes, as in the paper. IC and IRIC come within 2 points of success and 0.1 steps of the paper; their Traffic Junction success barely changes during training.
- On medium Traffic Junction, IC3Net exceeds the paper, while 19 of the 20 CommNet seeds stay at the level without communication. The reference code doesn't reach the paper's CommNet either: with the paper's curriculum, it gets 13.4. CommNet trained on each agent's own return (`MEAN_RATIO=0`), as IC3Net is, reaches 50% on all 20 seeds; the reference trains CommNet on the team's return. On medium Predator-Prey, CommNet and IC3Net take 1–4 more steps than in the paper.
- On hard Traffic Junction, neither this code nor the reference code reaches the paper's values at the paper's settings. This code is within 1.1 points of the reference code for IC, IRIC and CommNet, and above it for IC3Net. Cars can't see each other, so IC and IRIC score what cars that never brake score. CommNet with `MEAN_RATIO=0` reaches 71.8%, above the paper's CommNet. On hard Predator-Prey, IC, IRIC and CommNet match the paper, and IC3Net improves the most, as in the paper, but takes 66 steps against its 52.

### Comparison with the reference code

The [reference code](https://github.com/IC3Net/IC3Net), run with the edits current PyTorch needs, and this code trained Traffic Junction with identical settings for 1,000 epochs, half the paper's budget. Values are success (%), as the mean ± standard deviation over seeds of the last 2% of epochs.

Medium, CommNet, with the paper's curriculum and the team's return:

| Learning rate | Reference code (5 seeds) | This code (20 seeds) |
| --- | --- | --- |
| 0.003 | 13.4 ± 17.3 | 11.7 ± 19.8 |
| 0.001 | 9.2 ± 11.0 | 9.5 ± 16.2 |

Hard, at the paper's settings (`add_rate` 0.05, 60 steps), with no curriculum and a learning rate of 0.003. The last column is this code minus the reference, with a bootstrap 95% confidence interval:

| Method | Reference code (5 seeds) | This code (20 seeds) | Difference |
| --- | --- | --- | --- |
| IC | 13.6 ± 2.2 | 13.7 ± 4.2 | +0.1 [-2.4, 2.6] |
| IRIC | 16.6 ± 0.2 | 16.3 ± 0.3 | -0.4 [-0.5, -0.2] |
| CommNet | 8.3 ± 7.3 | 9.5 ± 5.9 | +1.1 [-5.1, 7.2] |
| IC3Net | 37.2 ± 21.6 | 55.3 ± 17.8 | +18.1 [-0.3, 36.6] |

IRIC's seeds barely vary in either code, so the 0.4-point gap is outside the interval. IC3Net's reference seeds split: two reach 56.8 and 61.9, and three stay between 16.2 and 34.3.

MPE episode return (higher is better), which the paper doesn't report:

| Method | Simple Reference | Speaker-Listener (`PAD_SPACES`) | Simple Spread, continuous |
| --- | --- | --- | --- |
| IC | -15.8 ± 1.6 [-16.5, -15.1] | -9.0 ± 2.0 [-9.9, -8.2] | -12.4 ± 0.5 [-12.6, -12.2] |
| IRIC | -14.8 ± 3.1 [-16.3, -13.6] | -9.0 ± 2.0 [-9.9, -8.2] | -12.6 ± 0.6 [-12.8, -12.3] |
| CommNet | -19.3 ± 3.5 [-20.8, -17.9] | -23.5 ± 12.5 [-28.6, -18.1] | -18.0 ± 2.1 [-19.0, -17.2] |
| IC3Net | -18.6 ± 5.9 [-21.1, -16.2] | -11.4 ± 7.9 [-15.1, -8.6] | -12.6 ± 0.5 [-12.8, -12.4] |

Communication doesn't help on these tasks with these settings: CommNet scores below IC, and IC3Net is no better than IC. Trained for three times as long (`TOTAL_TIMESTEPS=2.4e8`), CommNet stays below IC on Speaker-Listener and Simple Spread and is no longer clearly below it on Simple Reference.

These results aren't compared with IPPO and MAPPO: their scripts report success and episode length only for greedy test episodes, while `ic3net.py` reports training episodes. IPPO and MAPPO configurations for medium Traffic Junction and Predator-Prey are in the environments' READMEs.

Each table cell is printed by the script, for example:

```bash
python baselines/learned-communication/ic3net.py +alg=ic3net_traffic_junction_medium BASELINE=ic3net NUM_SEEDS=20 SEED_CHUNK=10
```

The results were produced with XLA's default GPU kernels, which aren't deterministic, so a rerun gives different per-seed results; `XLA_FLAGS=--xla_gpu_deterministic_ops=true` makes reruns identical.

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

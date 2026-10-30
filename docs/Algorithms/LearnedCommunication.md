# Learned Communication

Pure JAX implementations of:

* CommNet ([Sukhbaatar et al., 2016](https://arxiv.org/abs/1605.07736))
* IC3Net ([Singh et al., 2019](https://arxiv.org/abs/1812.09755))
* IC and IRIC, the no-communication controls from the IC3Net paper

The agents learn to communicate through a differentiable channel in their network, rather than through message actions in the environment. `ic3net.py` follows the IC3Net [reference implementation](https://github.com/IC3Net/IC3Net), which trains all four with REINFORCE and a learned value baseline.

Supported environments:

- Traffic Junction (`traffic_junction_grid`): the paper's easy, medium and hard versions
- Predator-Prey (`predator_prey_grid`): the paper's easy, medium and hard settings
- MPE Simple Reference
- MPE Speaker-Listener, whose agents have different observations and actions, with `PAD_SPACES` (`ic3net_mpe_speaker_listener`)
- MPE Simple Spread with continuous actions (`ic3net_mpe_simple_spread_continuous`)

All agents share parameters, so other environments need agents with the same observation and action spaces, or `PAD_SPACES=True`, and action spaces that are all discrete or all continuous. Their optional `alive`, `completed` and `success` info entries must hold one value per agent. Every agent communicates with every other and, for IC and CommNet, is trained on the mean return of all of them, so the script doesn't separate competing teams. In two-update smoke tests the script also runs, untuned and without results, on MPE Simple and Simple Spread, Switch Riddle, SMAX, STORM (`storm_np`), Hanabi, Overcooked, Overcooked V2, Coin Game and JaxNav with discrete actions. It doesn't use the action masks of Hanabi and SMAX.

## ⚙️ Implementation Details

- CommNet: before each LSTM step, every agent adds the mean of the other agents' hidden states to its input. CommNet agents always communicate.
- IC3Net: CommNet where each agent also chooses, at every step, whether to communicate at the next one, so no IC3Net agent communicates at the first step. Agents that are silent or dead (`info["alive"]`, when the environment reports it) neither send nor receive messages.
- IC and IRIC: independent recurrent agents, trained on the team's mean return (IC) or on their own (IRIC).
- Each update trains on whole episodes, with Monte Carlo returns and RMSprop (`alpha=0.97`, `eps=1e-6`) without gradient clipping, as in the reference. Losses are summed over agents and averaged over environment steps, and each step's loss is masked with the agents alive after it.
- With `MEAN_RATIO` below 1, an agent's own return ends when its episode does: a car that leaves Traffic Junction does not inherit the rewards of the next car in its slot.
- Reported statistics come from the training episodes.

`tests/baselines/test_ic3net.py` checks the returns, one step of the policies, and the loss and its gradients over a trajectory against transliterations of the reference's loops, as well as the rollout's timing of the alive and talk inputs, and the optimizer.

Differences from the reference:

- Layers use the orthogonal initialisation of the other JaxMARL baselines, and the LSTM flax's defaults, rather than PyTorch's defaults. Computation is in float32; the reference uses float64.
- Every update runs `NUM_ENVS` episodes of `NUM_STEPS` steps, 8,000 steps in the configurations (7,980 on hard Traffic Junction). The reference instead collects at least 500 steps of whole episodes in each of 16 processes: 8,000 to 8,960 steps per update when episodes last until the time limit, as in Traffic Junction. Predator-Prey episodes end once every predator reaches the prey, so as the agents improve, the reference's batch stays near 8,000 steps while the configurations' shrinks.
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
| `MEAN_RATIO` | Weight of the team's mean return in each agent's return. By default 1 for `ic` and `commnet` and 0 for `iric` and `ic3net`, as in the reference |
| `NUM_ENVS`, `NUM_STEPS` | Each update runs `NUM_ENVS` episodes from reset for `NUM_STEPS` steps; steps after an episode ends are ignored |
| `COMM_PASSES` | Communication rounds per step |
| `COMM_ACTION_ONE` | IC3Net agents always communicate; the reference sets it for Traffic Junction |
| `DETACH_GAP` | Steps of backpropagation through time |
| `SEED_CHUNK` | Trains the `NUM_SEEDS` seeds this many at a time, to fit in memory, with the same seeds as all at once |
| `LSTM` | By default, IC and IRIC use an LSTM, as in the paper; `False` gives the tanh RNN of the reference's commands. CommNet and IC3Net always use an LSTM |
| `PAD_SPACES` | Pads every agent's observation to the largest, followed by a one-hot agent id, and every agent's action space to the largest, masking the actions or action dimensions it doesn't have. Needed when agents have different spaces |
| `ACTION_SCALE` | For continuous actions, which use a Gaussian policy as in the reference: samples are multiplied by it, clipped to [-1, 1] and mapped onto the action space's bounds |

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

No other JaxMARL baseline trains with a curriculum, so Traffic Junction trains at the environment's fixed `add_rate`, as IPPO and MAPPO do; [with the curriculum](#with-the-curriculum), the gaps to the paper remain. The environments' READMEs list the other places where the paper's settings disagree with each other or with the reference.

## 📈 Results

All four methods with the configurations above, 20 seeds each, trained for 20,000 updates of 8,000 simulated steps on Traffic Junction (the paper's 2,000 epochs) and 10,000 on Predator-Prey (its 1,000). Values are the mean ± sample standard deviation over seeds of the training episodes in the last 2% of updates, followed by the bootstrap 95% confidence interval of the mean. The paper's values are over 5 runs; it doesn't say what its ± is.

Traffic Junction success (%):

| Method | Easy | Paper | Medium | Paper | Hard | Paper |
| --- | --- | --- | --- | --- | --- | --- |
| IC | 28.2 ± 0.1 [28.1, 28.2] | 30.2 ± 0.4 | 3.3 ± 0.2 [3.2, 3.4] | 3.4 ± 0.5 | 14.1 ± 2.0 [13.2, 14.9] | 47.0 ± 2.9 |
| IRIC | 28.2 ± 0.1 [28.1, 28.2] | 29.8 ± 0.7 | 3.4 ± 0.1 [3.4, 3.4] | 3.4 ± 0.5 | 16.3 ± 0.2 [16.2, 16.4] | 35.0 ± 0.6 |
| CommNet | 89.4 ± 15.7 [81.8, 94.5] | 93.0 ± 4.2 | 6.4 ± 13.2 [3.3, 12.3] | 54.3 ± 14.2 | 9.0 ± 5.0 [6.8, 11.2] | 50.2 ± 3.5 |
| IC3Net | 86.6 ± 5.9 [84.0, 89.0] | 93.0 ± 3.7 | 92.8 ± 5.1 [90.5, 94.8] | 89.3 ± 2.5 | 49.7 ± 24.2 [38.9, 59.9] | 72.4 ± 9.6 |

Predator-Prey steps to reach the prey:

| Method | Easy | Paper | Medium | Paper | Hard | Paper |
| --- | --- | --- | --- | --- | --- | --- |
| IC | 16.4 ± 0.4 [16.2, 16.5] | 16.4 ± 0.49 | 27.7 ± 0.7 [27.4, 28.0] | 28.0 ± 0.74 | 76.1 ± 1.7 [75.4, 76.8] | 77.4 ± 0.8 |
| IRIC | 16.4 ± 0.6 [16.1, 16.6] | 16.5 ± 0.1 | 28.2 ± 0.4 [28.0, 28.4] | 28.1 ± 0.2 | 75.3 ± 1.5 [74.7, 75.9] | 75.0 ± 1.4 |
| CommNet | 9.4 ± 0.1 [9.3, 9.4] | 9.1 ± 0.1 | 14.4 ± 1.5 [13.9, 15.2] | 13.1 ± 0.01 | 76.5 ± 1.9 [75.7, 77.3] | 76.5 ± 1.3 |
| IC3Net | 10.2 ± 2.2 [9.3, 11.2] | 8.9 ± 0.02 | 16.7 ± 2.1 [15.9, 17.7] | 13.0 ± 0.02 | 64.3 ± 7.4 [61.1, 67.4] | 52.4 ± 3.4 |

- On the easy versions, communication raises Traffic Junction success from 28% to 87–89% and nearly halves the steps to the prey, as in the paper. IC and IRIC come within 2 points of success and 0.1 steps of the paper; on Traffic Junction their success barely changes during training (28% from the start), so the match there reflects car arrivals more than learning. CommNet and IC3Net fall 4–6 points short on Traffic Junction; one CommNet seed stays at 28%. IC and IRIC end with the same easy Traffic Junction success because both converge to near-deterministic policies, and runs with the same seed get the same car arrivals.
- On medium Traffic Junction, IC3Net exceeds the paper, with all 20 seeds at 79–98% success, while 19 CommNet seeds stay at 2.4–4.0%, the level without communication.
- On medium Predator-Prey, CommNet and IC3Net take 1–4 more steps than in the paper. In a run with three times as many episodes per update (`alg.NUM_ENVS=600`) for 20,000 updates, on an earlier version of `ic3net.py`, 4 seeds of CommNet end at 13.08 ± 0.03 steps, the paper's 13.1, but 4 seeds of IC3Net end at 15.08 ± 0.91.
- On hard Traffic Junction, all four methods fall well short of the paper. IC3Net still leads, but five of its seeds end at 13–17% success, the level without communication.
- On hard Predator-Prey, IC, IRIC and CommNet match the paper, taking 75–77 of the 80 allowed steps. IC3Net is the only method that improves much, as in the paper, but takes 64 steps against the paper's 52.
- For comparison, IPPO and MAPPO on the same medium environments (see the environments' READMEs) reach 3–5% success on Traffic Junction and take 26 steps on Predator-Prey, after 2e7 steps. Those are greedy test episodes, while the results here are training episodes.

### With the curriculum

The same 20 seeds of each method trained on Traffic Junction with the paper's curriculum restored, using `ic3net.py` and its configurations as of commit 2b0f64b, with the settings above: `LSTM=True alg.LR=0.003`, and on hard also `+alg.ENV_KWARGS.max_steps=60 alg.NUM_ENVS=133 alg.NUM_STEPS=60 alg.TOTAL_TIMESTEPS=1.596e8`. Every run had reached the fixed `add_rate` by the end, so the success (%) below is on the same task as above:

| Method | Easy | Medium | Hard |
| --- | --- | --- | --- |
| IC | 28.2 ± 0.1 [28.1, 28.2] | 12.9 ± 10.5 [8.5, 17.5] | 14.6 ± 2.8 [13.3, 15.7] |
| IRIC | 28.2 ± 0.1 [28.1, 28.2] | 3.4 ± 0.1 [3.4, 3.4] | 16.3 ± 0.2 [16.2, 16.4] |
| CommNet | 89.5 ± 8.4 [85.9, 93.0] | 21.7 ± 27.4 [10.9, 34.1] | 12.0 ± 4.7 [9.9, 13.9] |
| IC3Net | 89.7 ± 6.6 [86.7, 92.4] | 79.8 ± 16.1 [72.4, 86.1] | 35.5 ± 21.5 [26.6, 44.7] |

The curriculum doesn't close the gaps to the paper. Only two changes are larger than the confidence intervals: on medium, it lowers IC3Net from 92.8 to 79.8, and raises IC from 3.3 to 12.9, above the paper's 3.4. CommNet on medium (6.4 to 21.7) and IC3Net on hard (49.7 to 35.5) change within overlapping intervals.

### MPE Simple Reference

`ic3net_mpe_simple_reference` trains on `MPE_simple_reference_v3`, which the paper doesn't use, with the same batch size and number of updates as Predator-Prey. Each of the two agents sees the landmark the other must reach. Episode return (higher is better):

| Method | Return |
| --- | --- |
| IC | -15.8 ± 1.6 [-16.5, -15.1] |
| IRIC | -14.8 ± 3.1 [-16.3, -13.6] |
| CommNet | -19.3 ± 3.5 [-20.8, -17.9] |
| IC3Net | -17.6 ± 5.7 [-20.0, -15.2] |

CommNet and IC3Net end lower on average because some of their seeds don't learn: 3 CommNet seeds, 5 IC3Net seeds and 1 IRIC seed end near -26, where training starts, against none of IC's. IC3Net's best seeds (-10.7) end above IC's best (-12.1). All four methods were still improving at the end, and the configuration isn't tuned for this environment.

### MPE with padding and continuous actions

`ic3net_mpe_speaker_listener` and `ic3net_mpe_simple_spread_continuous` use the same, untuned, settings as `ic3net_mpe_simple_reference`. Episode return at the end of training, with the return in the first 2% of updates in brackets:

| Method | Speaker-Listener (`PAD_SPACES`) | Simple Spread, continuous |
| --- | --- | --- |
| IC | -9.0 ± 2.0 [-9.9, -8.2] (-23.0) | -12.5 ± 0.5 [-12.7, -12.3] (-22.3) |
| IRIC | -9.0 ± 2.0 [-9.9, -8.2] (-23.0) | -12.6 ± 0.6 [-12.8, -12.3] (-22.5) |
| CommNet | -23.5 ± 12.5 [-28.6, -18.1] (-33.6) | -18.0 ± 2.1 [-19.0, -17.2] (-24.7) |
| IC3Net | -11.4 ± 7.9 [-15.1, -8.6] (-25.6) | -12.6 ± 0.5 [-12.8, -12.4] (-22.9) |

- Every method improves on both tasks. CommNet ends lowest on both, as on Simple Reference; on Speaker-Listener, some of its seeds end near where training starts (worst seed -33.5).
- IC and IRIC end identically on Speaker-Listener, where both agents receive the same reward, so the team's mean return and an agent's own are the same.
- IC3Net agents communicate at 14% of steps on Speaker-Listener and 1% on Simple Spread.

### Reproducing the results

Wall time for 20 seeds on one L40S, including compilation, with `SEED_CHUNK=10` (5 on hard):

| Method | Traffic Junction easy | medium | hard | Predator-Prey easy | medium | hard | MPE Simple Reference |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IC, IRIC | 26 min | 56 min | 132 min | 8 min | 28 min | 203 min | 5–6 min |
| CommNet, IC3Net | 28–30 min | 61–64 min | 143–148 min | 8–9 min | 31 min | 206–207 min | 5–6 min |

Speaker-Listener takes 5–6 minutes and continuous Simple Spread 7.5–9.

These commands train the same seeds, for each `BASELINE`, and print the statistics of the tables. The tables were produced with XLA's default GPU kernels, which aren't deterministic, so a rerun with the same seeds ends with different per-seed results. `XLA_FLAGS=--xla_gpu_deterministic_ops=true` makes reruns identical: two runs of 4 seeds of medium Traffic Junction for 200 updates had identical losses with it and diverged within the first 100 updates without it.

```bash
python baselines/learned-communication/ic3net.py +alg=ic3net_traffic_junction_medium BASELINE=ic3net NUM_SEEDS=20 SEED_CHUNK=10
python baselines/learned-communication/ic3net.py +alg=ic3net_predator_prey_medium BASELINE=ic3net NUM_SEEDS=20 SEED_CHUNK=10
python baselines/learned-communication/ic3net.py +alg=ic3net_traffic_junction_hard BASELINE=ic3net NUM_SEEDS=20 SEED_CHUNK=5
python baselines/learned-communication/ic3net.py +alg=ic3net_mpe_simple_reference BASELINE=ic3net NUM_SEEDS=20 SEED_CHUNK=10
python baselines/learned-communication/ic3net.py +alg=ic3net_mpe_speaker_listener BASELINE=ic3net NUM_SEEDS=20 SEED_CHUNK=10
python baselines/learned-communication/ic3net.py +alg=ic3net_mpe_simple_spread_continuous BASELINE=ic3net NUM_SEEDS=20 SEED_CHUNK=10
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

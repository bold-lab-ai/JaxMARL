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

The configurations are the IC3Net paper's easy, medium and hard settings of `traffic_junction_grid` and `predator_prey_grid`, with the paper's training settings but without its curriculum (see [Settings](#settings)). All agents share parameters, so other environments need homogeneous agents and discrete actions, and their optional `alive`, `completed` and `success` info entries must hold one value per agent; the script has only been trained on these two.

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

## ⚙️ Implementation details

The tests in `tests/baselines/test_ic3net.py` check the returns, one step of the policies, and the loss and its gradients over a trajectory against transliterations of the reference's loops. They also check the rollout's timing of the alive and talk inputs, and the optimizer. As in the reference:

- Each update trains on whole episodes, with Monte Carlo returns and RMSprop (`alpha=0.97`, `eps=1e-6`) without gradient clipping. Losses are summed over agents and averaged over environment steps.
- With `MEAN_RATIO` below 1, an agent's own return ends when its episode does: a car that leaves Traffic Junction does not inherit the rewards of the next car in its slot.
- Agents that are dead (`info["alive"]`, when the environment reports it) or silent neither send nor receive messages. An IC3Net agent's choice to communicate takes effect at the next step, so no IC3Net agent communicates at the first; CommNet agents always communicate.
- Each step's loss is masked with the agents alive after it. A car's final step on the road is therefore not trained, and a car that enters during a step is trained on that step, though it acted on an empty observation.
- Reported statistics come from the training episodes.

Differences from the reference:

- Layers use the orthogonal initialisation of the other JaxMARL baselines, and the LSTM flax's defaults, rather than PyTorch's defaults. Computation is in float32; the reference uses float64.
- Every update runs a fixed number of episodes; the reference runs 16 processes that each collect at least 500 steps of whole episodes, which is 8,000 steps per update on easy, 8,320 on medium and 8,960 on hard (8,640 on hard Traffic Junction, with 60 steps) when episodes last until the time limit. The configurations run 8,000 (7,980 on hard Traffic Junction). Traffic Junction episodes always last until the time limit, but Predator-Prey episodes end once every predator reaches the prey, so as the agents improve, the batch shrinks well below the reference's.
- The entropy term is masked like the rest of the loss; the reference sums it over all agents. It is unused with the default `ENT_COEF` of 0.
- There is no curriculum (see [Settings](#settings)).

### Settings

The configurations follow the paper's training settings, and the reference's commands where the paper doesn't give one, except for the curriculum:

| Setting | Paper | Reference's commands | Configurations |
| --- | --- | --- | --- |
| Traffic Junction learning rate | 0.003 | 0.001 | 0.003 |
| Predator-Prey learning rate | Tuned, not given | 0.001 | 0.001 |
| Predator-Prey training | 1,000 epochs | 2,000 epochs | 1,000 epochs |
| IC and IRIC network | LSTM | tanh RNN | LSTM; tanh RNN with `LSTM=False` |
| Hard Traffic Junction `max_steps` | 60 | 80 | 60 |
| Traffic Junction `add_rate` | Raised from 0.1, 0.05 and 0.02 to 0.3, 0.2 and 0.05 | The same, but medium's curriculum is disabled | Fixed at 0.3, 0.2 and 0.05 |

No other JaxMARL baseline trains with a curriculum, so Traffic Junction trains at the environment's fixed `add_rate`, as IPPO and MAPPO do. Its results are therefore not the paper's experiment. The paper and the configurations both train Traffic Junction for 2,000 epochs of 10 updates. The environments' READMEs list the other places where the paper's settings disagree with each other or with the reference.

## 📈 Results

All four methods with the configurations above: 20 seeds each, trained for 20,000 updates of 8,000 simulated steps on Traffic Junction (the paper's 2,000 epochs) and 10,000 on Predator-Prey (its 1,000). Values are the mean ± sample standard deviation over seeds of the training episodes in the last 2% of updates, followed by the bootstrap 95% confidence interval of the mean. The paper's values are over 5 runs; it doesn't say what its ± is.

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

- On the easy versions, communication raises Traffic Junction success from 28% to 87–89% and nearly halves the steps to the prey, as in the paper. Without communication, IC and IRIC come within 2 points of success and 0.1 steps of the paper. CommNet and IC3Net fall 4–6 points short of the paper on Traffic Junction. 18 CommNet seeds reach 89–99% success, one 72%, and one stays at 28%, the level without communication. On Predator-Prey, 17 IC3Net seeds take 9.2–9.6 steps and three take 12.7–16.2.
- IC and IRIC end with the same easy Traffic Junction success: both converge to near-deterministic policies, and runs with the same seed get the same car arrivals.
- On medium Traffic Junction, without the curriculum, IC3Net exceeds the paper: all 20 seeds end at 79–98% success. Its mean success stays at 3.4% for the first tenth of training, reaches 88% half way, and 93% at the end. 19 CommNet seeds stay at 2.4–4.0%, the level without communication, and one reaches 62%. IC and IRIC match the paper.
- On medium Predator-Prey, IC and IRIC match the paper. 16 CommNet seeds take 13.5–14.4 steps, against the paper's 13.1, and four take 15.2–20.2. IC3Net doesn't overtake CommNet; its agents communicate at 72% of steps. Three times as many episodes per update help CommNet: with `alg.NUM_ENVS=600 alg.TOTAL_TIMESTEPS=2.4e8` (the same number of updates, and batches near the reference's size once episodes are short), 4 seeds of CommNet take 13.2 ± 0.1 [13.2, 13.3] steps. 4 seeds of IC3Net take 15.8 ± 0.6 [15.4, 16.4], within the spread of its 20 seeds, so that run doesn't settle whether batch size explains IC3Net's gap.
- On hard Traffic Junction, all four methods fall well short of the paper. IC3Net still leads, but five of its seeds end at 13–17% success, the level without communication, and twelve reach 57–78%; its mean success stopped rising after three quarters of training. CommNet ends below IC and IRIC.
- On hard Predator-Prey, IC, IRIC and CommNet match the paper, taking 75–77 of the 80 allowed steps. IC3Net is the only method that improves much, as in the paper, but takes 64 steps against the paper's 52; its agents communicate at 71% of steps.
- For comparison, IPPO and MAPPO on the same medium environments (see the environments' READMEs) reach 3–5% success on Traffic Junction and take 26 steps on Predator-Prey. They train for 2e7 steps and report greedy test episodes.

Wall time for 20 seeds on one L40S, including compilation, with `SEED_CHUNK=10` (5 on hard):

| Method | Traffic Junction easy | medium | hard | Predator-Prey easy | medium | hard |
| --- | --- | --- | --- | --- | --- | --- |
| IC, IRIC | 26 min | 56 min | 132 min | 8 min | 28 min | 203 min |
| CommNet, IC3Net | 28–30 min | 61–64 min | 143–148 min | 8–9 min | 31 min | 206–207 min |

These commands train the same seeds, for each `BASELINE`, and print the statistics of the tables; the results were produced by a script that calls the same `make_train` and also saved the per-update metrics:

```bash
python baselines/learned-communication/ic3net.py +alg=ic3net_traffic_junction_medium BASELINE=ic3net NUM_SEEDS=20 SEED_CHUNK=10
python baselines/learned-communication/ic3net.py +alg=ic3net_predator_prey_medium BASELINE=ic3net NUM_SEEDS=20 SEED_CHUNK=10
python baselines/learned-communication/ic3net.py +alg=ic3net_traffic_junction_hard BASELINE=ic3net NUM_SEEDS=20 SEED_CHUNK=5
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

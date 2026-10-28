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

The configurations are the IC3Net paper's easy, medium and hard settings of `traffic_junction_grid` and `predator_prey_grid`, with the training settings of the reference's commands (see [Settings](#settings)). All agents share parameters, so other environments need homogeneous agents and discrete actions, and their optional `alive`, `completed` and `success` info entries must hold one value per agent; the script has only been trained on these two.

| Key | Meaning |
| --- | --- |
| `BASELINE` | `ic`, `iric`, `commnet` or `ic3net` |
| `MEAN_RATIO` | Weight of the team's mean return in each agent's return. By default 1 for `ic` and `commnet` and 0 for `iric` and `ic3net`, as in the reference |
| `NUM_ENVS`, `NUM_STEPS` | Each update runs `NUM_ENVS` episodes from reset for `NUM_STEPS` steps; steps after an episode ends are ignored |
| `COMM_PASSES` | Communication rounds per step |
| `COMM_ACTION_ONE` | IC3Net agents always communicate; the reference sets it for Traffic Junction |
| `DETACH_GAP` | Steps of backpropagation through time |
| `CURRICULUM` | Optional. Raises the environment argument `KEY` from `START` to `END` in steps of `STEP`, between the fractions `BEGIN` and `FINISH` of training. The Traffic Junction configurations raise `add_rate` as the paper does |
| `SEED_CHUNK` | Trains the `NUM_SEEDS` seeds this many at a time, to fit in memory, with the same seeds as all at once |

## ⚙️ Implementation details

The tests in `tests/baselines/test_ic3net.py` check the returns, one step of the policies, and the loss and its gradients over a trajectory against transliterations of the reference's loops. They also check the rollout's timing of the alive and talk inputs, the optimizer and the curriculum. As in the reference:

- Each update trains on whole episodes, with Monte Carlo returns and RMSprop (`alpha=0.97`, `eps=1e-6`) without gradient clipping. Losses are summed over agents and averaged over environment steps.
- With `MEAN_RATIO` below 1, an agent's own return ends when its episode does: a car that leaves Traffic Junction does not inherit the rewards of the next car in its slot.
- Agents that are dead (`info["alive"]`, when the environment reports it) or silent neither send nor receive messages. An IC3Net agent's choice to communicate takes effect at the next step, so no IC3Net agent communicates at the first; CommNet agents always communicate.
- Each step's loss is masked with the agents alive after it. A car's final step on the road is therefore not trained, and a car that enters during a step is trained on that step, though it acted on an empty observation.
- Traffic Junction trains with the paper's curriculum: `add_rate` starts at the paper's lower value and rises in steps of 0.01 between epochs 250 and 1,250 of 2,000. In short pilot runs without it, on an earlier version of this script (2 seeds, 2e7 steps), all four methods stayed at about 3.4% success.
- Reported statistics come from the training episodes.

Differences from the reference:

- Layers use the orthogonal initialisation of the other JaxMARL baselines, and the LSTM flax's defaults, rather than PyTorch's defaults. Computation is in float32; the reference uses float64.
- Every update runs a fixed number of episodes; the reference runs 16 processes that each collect at least 500 steps of whole episodes, which is 8,000 steps per update on easy, 8,320 on medium and 8,960 on hard when episodes last until the time limit. The configurations run 8,000. Traffic Junction episodes always last until the time limit, but Predator-Prey episodes end once every predator reaches the prey, so as the agents improve, the batch shrinks well below the reference's.
- The entropy term is masked like the rest of the loss; the reference sums it over all agents. It is unused with the default `ENT_COEF` of 0.
- The curriculum changes level one epoch later than the reference's, and on easy it reaches 0.3, where floating-point rounding stops the reference's at 0.29.

### Settings

Where the paper and the reference's training commands disagree, the configurations follow the commands, except for the medium Traffic Junction curriculum:

| Setting | Paper | Reference's commands | Configurations |
| --- | --- | --- | --- |
| Traffic Junction learning rate | 0.003 | 0.001 | 0.001 |
| Predator-Prey training | 1,000 epochs | 2,000 epochs | 2,000 epochs |
| IC and IRIC network | LSTM | tanh RNN | tanh RNN |
| Hard Traffic Junction `max_steps` | 60 | 80 | 80 |
| Medium Traffic Junction `add_rate` | 0.05 to 0.2 | 0.05 to 0.02, which disables the curriculum | 0.05 to 0.2 |

The environments' READMEs list the other places where the paper's settings disagree with each other or with the reference.

## 📈 Results

All four methods on the medium versions, with the configurations above: 20 seeds each, trained for 20,000 updates of 8,000 simulated steps (the reference's 2,000 epochs; the paper trains Predator-Prey for 1,000). Values are the mean ± standard deviation over seeds of the training episodes in the last 2% of updates. The paper's values are over 5 runs; it doesn't say what its ± is.

| Method | Traffic Junction success (%) | Paper | Predator-Prey steps | Paper |
| --- | --- | --- | --- | --- |
| IC | 4.7 ± 6.0 | 3.4 ± 0.5 | 28.0 ± 0.4 | 28.0 ± 0.74 |
| IRIC | 3.4 ± 0.1 | 3.4 ± 0.5 | 28.3 ± 0.3 | 28.1 ± 0.2 |
| CommNet | 30.7 ± 27.6 | 54.3 ± 14.2 | 13.8 ± 1.4 | 13.1 ± 0.01 |
| IC3Net | 67.0 ± 26.8 | 89.3 ± 2.5 | 16.3 ± 1.5 | 13.0 ± 0.02 |

- Without communication, IC and IRIC match the paper on both tasks, except for one IC seed that reaches 30.9% success on Traffic Junction; the other 39 of their 40 seeds end at 3.2–3.5%.
- On Traffic Junction, the methods rank as in the paper, but both communicating methods fall short of it, with a lower learning rate than the paper's (see [Settings](#settings)). IC3Net varies far more across seeds than the paper's ± suggests: eight seeds exceed 85% success and three stay below 25%, and its mean success was still rising at the end of training. Eight CommNet seeds stay below 6%, four reach 11–26% and eight reach 46–74%.
- On Predator-Prey, CommNet comes close to the paper (19 of 20 seeds take 13.3–13.6 steps, one 19.8), but IC3Net does not overtake it; its agents communicate at 70% of steps. Three times as many episodes per update help CommNet: with `alg.NUM_ENVS=600 alg.TOTAL_TIMESTEPS=4.8e8` (the same number of updates, and batches near the reference's size once episodes are short), 4 seeds of CommNet take 13.1 ± 0.03 steps. 4 seeds of IC3Net take 15.1 ± 0.8, within the spread of its 20 seeds, so that run doesn't settle whether batch size explains IC3Net's gap.
- For comparison, IPPO and MAPPO on the same environments (see the environments' READMEs) reach 3–5% success on Traffic Junction and take 26 steps on Predator-Prey. They train for 2e7 steps, without the curriculum, and report greedy test episodes.

Wall time for 20 seeds on one L40S, with `SEED_CHUNK=10`, including compilation, and the simulated steps per second summed over seeds:

| Method | Traffic Junction | Predator-Prey | Simulated steps per second |
| --- | --- | --- | --- |
| IC, IRIC | 18 min | 40 min | 3.0e6 (TJ), 1.3e6 (PP) |
| CommNet, IC3Net | 61–64 min | 62 min | 8.3e5–8.7e5 |

These commands train the same seeds, for each `BASELINE`, and print the statistics of the table; the results were produced by a script that calls the same `make_train` and also saved the per-update metrics:

```bash
python baselines/learned-communication/ic3net.py +alg=ic3net_traffic_junction_medium BASELINE=ic3net NUM_SEEDS=20 SEED_CHUNK=10
python baselines/learned-communication/ic3net.py +alg=ic3net_predator_prey_medium BASELINE=ic3net NUM_SEEDS=20 SEED_CHUNK=10
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

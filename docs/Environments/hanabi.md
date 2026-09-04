# Hanabi

This directory contains a MARL environment for the cooperative card game, Hanabi, implemented in JAX. It is inspired by the popular [Hanabi Learning Environment (HLE)](https://arxiv.org/pdf/1902.00506.pdf), but intended to be simpler to integrate and run with the growing ecosystem of JAX implemented RL research pipelines. 


## Action Space
Hanabi is a turn-based game. The current player can choose to discard or play any of the cards in their hand, or hint a colour or rank to any one of their teammates.

## Configuration

Set through the environment constructor:

| Argument | Default | Meaning |
| --- | --- | --- |
| `num_agents` | `2` | Players (2-5) |
| `num_colors` | `5` | Suits |
| `num_ranks` | `5` | Ranks per suit |
| `num_cards_of_rank` | `[3, 2, 2, 2, 1]` | Copies of each rank within a suit; one entry per rank |
| `color_map` | first `num_colors` of `["R", "Y", "G", "W", "B", "M"]` | One-character suit label, used for rendering and hint action names |
| `hand_size` | `5` (2-3 players), `4` (4-5) | Cards per hand |
| `max_info_tokens` | `8` | Clue tokens |
| `max_life_tokens` | `3` | Lives |

### Six suits

```python
env = make("hanabi", num_colors=6)                              # sixth suit is "M"
env = make("hanabi", num_colors=6, color_map=list("RYGWBP"))    # relabelled
```

The physical game's multicolor suit, played as an ordinary colour: a full
`3/2/2/2/1` set of cards for a 60-card deck, and a maximum score of 30. Colour
hints are indexed by (teammate, colour), so it costs `num_agents - 1` extra
actions — one "hint M" per teammate, taking a 2-player game from 21 to 22.

The *rainbow* rules, where a multicolor card is touched by every colour clue,
are a separate variant and are not implemented.

## Observation Space
The observations closely follow the featurization in the HLE. Each observation is comprised of 658 features:

* **Hands (127)**: information about the visible hands.
  * other player hand: 125 
    * card 0: 25,
    * card 1: 25
    * card 2: 25
    * card 3: 25
    * card 4: 25
  * Hands missing card: 2 (one-hot)

* **Board (76)**: encoding of the public information visible in the board.
  * Deck: 40, thermometer 
  * Fireworks: 25, one-hot
  * Info Tokens: 8, thermometer
  * ife Tokens: 3, thermometer

* **Discards (50)**: encoding of the cards in the discard pile.
  * Colour R: 10 bits for each card
  * Colour Y: 10 bits for each card
  * Colour G: 10 bits for each card
  * Colour W: 10 bits for each card
  * Colour B: 10 bits for each card

* **Last Action (55)**: encoding of the last move of the previous player.
  * Acting player index, relative to yourself: 2, one-hot
  * MoveType: 4, one-hot
  * Target player index, relative to acting player: 2, one-hot
  * Color revealed: 5, one-hot
  * Rank revealed: 5, one-hot
  * Reveal outcome 5 bits, each bit is 1 if the card was hinted at
  * Position played/discarded: 5, one-hot
  * Card played/discarded 25, one-hot
  * Card played scored: 1
  * Card played added info token: 1

* **V0 belief (350)**: trivially-computed probability of being a specific car (given the played-discarded cards and the hints given), for each card of each player.
  * Possible Card (for each card): 25 (* 10)
  * Colour hinted (for each card): 5 (* 10)
  * Rank hinted (for each card): 5 (* 10)

The counts above are for the default 5-suit deck; every block scales with the configuration. With `num_colors=6` the observation is 774 features: Hands 152, Board 91, Discards 60, Last Action 61, V0 belief 410.

## Pretrained Models

We make available to use some pretrained models. For example you can use a jax conversion of the original R2D2 OBL model in this way:

1. Download the models from Hugginface: ```git clone https://huggingface.co/mttga/obl-r2d2-flax``` (ensure to have git lfs installed). You can also use the script: ```bash jaxmarl/environments/hanabi/models/download_r2d2_obl.sh```
2. Load the parameters, import the agent wrapper and use it with JaxMarl Hanabi:

```python
!git clone https://huggingface.co/mttga/obl-r2d2-flax
import jax
from jax import numpy as jnp
from jaxmarl import make
from jaxmarl.wrappers.baselines import load_params
from jaxmarl.environments.hanabi.pretrained import OBLAgentR2D2

weight_file = "jaxmarl/environments/hanabi/pretrained/obl-r2d2-flax/icml_OBL1/OFF_BELIEF1_SHUFFLE_COLOR0_BZA0_BELIEF_a.safetensors"
params = load_params(weight_file)

agent = OBLAgentR2D2()
agent_carry = agent.initialize_carry(jax.random.PRNGKey(0), batch_dims=(2,))

rng = jax.random.PRNGKey(0)
env = make('hanabi')
obs, env_state = env.reset(rng)
env.render(env_state)

batchify = lambda x: jnp.stack([x[agent] for agent in env.agents])
unbatchify = lambda x: {agent:x[i] for i, agent in enumerate(env.agents)}

agent_input = (
    batchify(obs),
    batchify(env.get_legal_moves(env_state))
)
agent_carry, actions = agent.greedy_act(params, agent_carry, agent_input)
actions = unbatchify(actions)

obs, env_state, rewards, done, info = env.step(rng, env_state, actions)

print('actions:', {agent:env.action_encoding[int(a)] for agent, a in actions.items()})
env.render(env_state)
```

## Rendering

You can render the full environment state:

```python
obs, env_state = env.reset(rng)
env.render(env_state)

Turn: 0

Score: 0
Information: 8
Lives: 3
Deck: 40
Discards:                                                  
Fireworks:     
Actor 0 Hand:<-- current player
0 W3 || XX|RYGWB12345
1 G5 || XX|RYGWB12345
2 G4 || XX|RYGWB12345
3 G1 || XX|RYGWB12345
4 Y2 || XX|RYGWB12345
Actor 1 Hand:
0 R3 || XX|RYGWB12345
1 B1 || XX|RYGWB12345
2 G1 || XX|RYGWB12345
3 R4 || XX|RYGWB12345
4 W4 || XX|RYGWB12345
```

Or you can render the partial observation of the current agent:

```python
obs, new_env_state, rewards, dones, infos  = env.step_env(rng, env_state, actions)
obs_s = env.get_obs_str(new_env_state, env_state, a, include_belief=True, best_belief=5)
print(obs_s)

Turn: 1

Score: 0
Information available: 7
Lives available: 3
Deck remaining cards: 40
Discards:                                                  
Fireworks:     
Other Hand:
0 Card: W3, Hints: , Possible: RYGWB12345, Belief: [R1: 0.060 Y1: 0.060 G1: 0.060 W1: 0.060 B1: 0.060]
1 Card: G5, Hints: , Possible: RYGWB12345, Belief: [R1: 0.060 Y1: 0.060 G1: 0.060 W1: 0.060 B1: 0.060]
2 Card: G4, Hints: , Possible: RYGWB12345, Belief: [R1: 0.060 Y1: 0.060 G1: 0.060 W1: 0.060 B1: 0.060]
3 Card: G1, Hints: , Possible: RYGWB12345, Belief: [R1: 0.060 Y1: 0.060 G1: 0.060 W1: 0.060 B1: 0.060]
4 Card: Y2, Hints: , Possible: RYGWB12345, Belief: [R1: 0.060 Y1: 0.060 G1: 0.060 W1: 0.060 B1: 0.060]
Your Hand:
0 Hints: ,  Possible: RYGWB2345, Belief: [R2: 0.057 R3: 0.057 R4: 0.057 Y2: 0.057 Y3: 0.057]
1 Hints: 1, Possible: RYGWB1,    Belief: [R1: 0.200 Y1: 0.200 G1: 0.200 W1: 0.200 B1: 0.200]
2 Hints: 1, Possible: RYGWB1,    Belief: [R1: 0.200 Y1: 0.200 G1: 0.200 W1: 0.200 B1: 0.200]
3 Hints: ,  Possible: RYGWB2345, Belief: [R2: 0.057 R3: 0.057 R4: 0.057 Y2: 0.057 Y3: 0.057]
4 Hints: ,  Possible: RYGWB2345, Belief: [R2: 0.057 R3: 0.057 R4: 0.057 Y2: 0.057 Y3: 0.057]
Last action: H1
Cards afected: [1 2]
Legal Actions: ['D0', 'D1', 'D2', 'D3', 'D4', 'P0', 'P1', 'P2', 'P3', 'P4', 'HY', 'HG', 'HW', 'H1', 'H2', 'H3', 'H4', 'H5']
```

## Manual Game

You can test the environment and your models by using the ```manual_game.py``` script in this folder. It allows to control one or two agents with the keyboard and one or two agents with a pretrained model (an obl model by default). For example, to play with an obl pretrained model:

```
python manual_game.py \
  --player0 "manual" \
  --player1 "obl" \
  --weight1 "./pretrained/obl-r2d2-flax/icml_OBL1/OFF_BELIEF1_SHUFFLE_COLOR0_BZA0_BELIEF_a.safetensors" \
```

Or to look an obl model playing with itself:

```
python manual_game.py \
  --player0 "obl" \
  --player1 "obl" \
  --weight0 "./pretrained/obl-r2d2-flax/icml_OBL1/OFF_BELIEF1_SHUFFLE_COLOR0_BZA0_BELIEF_a.safetensors" \
  --weight1 "./pretrained/obl-r2d2-flax/icml_OBL1/OFF_BELIEF1_SHUFFLE_COLOR0_BZA0_BELIEF_a.safetensors" \
```

## Citation
The environment was orginally described in the following work:
```
@article{bard2019hanabi,
  title={The Hanabi Challenge: A New Frontier for AI Research},
  author={Bard, Nolan and Foerster, Jakob N. and Chandar, Sarath and Burch, Neil and Lactot, Marc and Song,    H. Francis and Parisotto, Emilio and Dumoulin, Vincent and Moitra, Subhodeep and Hughes, Edward and          Dunning, Ian and Mourad, Shibl and Larochelle, Hugo and Bellemare, Marc G. and Bowling},
  journal={Artificial Intelligence Journal},
  year={2019}
}
```

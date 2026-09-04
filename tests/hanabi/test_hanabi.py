import json
import os

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jaxmarl import make
from jaxmarl.environments.hanabi.hanabi import HanabiEnv

env = make("hanabi")
dir_path = os.path.dirname(os.path.realpath(__file__))


def pad_array(arr, target_length):
    pad_size = target_length - len(arr)
    if pad_size > 0:
        return np.pad(arr, (0, pad_size), "constant")
    else:
        return arr


def get_action_sequences():
    with open(os.path.join(dir_path, "actions.json"), "r") as f:
        actions_traj = json.load(f)

    max_len = max(len(a) for a in actions_traj)
    actions_traj = np.array([pad_array(np.array(a), max_len) for a in actions_traj])
    return jnp.array(actions_traj)


def get_decks():

    color_map = dict(zip(["R", "Y", "G", "W", "B"], range(5)))

    def gen_cards(color, rank):
        card = np.zeros((5, 5))
        card[color, rank] = 1
        return card

    with open(os.path.join(dir_path, "decks.json"), "r") as f:
        decks = json.load(f)

    # encode into card-matrices
    decks = np.array(
        [
            [gen_cards(color_map[card[0]], int(card[1]) - 1) for card in deck["deck"]]
            for deck in decks
        ]
    )

    return jnp.array(decks)


def get_scores():
    with open(os.path.join(dir_path, "scores.txt"), "r") as f:
        scores = [int(line.split(",")[1].split("\n")[0]) for line in f.readlines()]
    return jnp.array(scores)


def get_injected_score(deck, actions):

    def _env_step(env_state, action):

        curr_player = jnp.where(env_state.cur_player_idx == 1, size=1)[0][0]
        actions = jnp.array([20, 20]).at[curr_player].set(action)
        actions = {agent: action for agent, action in zip(env.agents, actions)}

        new_obs, new_env_state, reward, done, info = env.step(
            jax.random.PRNGKey(0), env_state, actions
        )
        return new_env_state, (reward, done)

    obs, env_state = env.reset_from_deck(deck)
    _, (rewards, dones) = jax.lax.scan(_env_step, env_state, actions)

    def first_episode_returns(rewards, dones):
        first_done = jax.lax.select(
            jnp.argmax(dones) == 0.0, dones.size, jnp.argmax(dones)
        )
        first_episode_mask = jnp.where(
            jnp.arange(dones.size) <= first_done, True, False
        )
        return jnp.where(first_episode_mask, rewards, 0.0).sum()

    cum_rewards = first_episode_returns(rewards["__all__"], dones["__all__"])
    return cum_rewards


def make_single_life_env():
    return HanabiEnv(
        num_agents=2,
        num_colors=5,
        num_ranks=5,
        hand_size=5,
        max_info_tokens=8,
        max_life_tokens=1,
    )


def single_life_loss_deck():
    # Player 0 starts with a rank-1 card in slot 0 while fireworks are empty, so play-0 is invalid.
    deck = np.zeros((50, 2), dtype=int)
    deck[:, 0] = 0
    deck[:, 1] = 0
    deck[0] = np.array([0, 1])
    return jnp.array(deck)


def single_life_actions(env):
    return {
        "agent_0": env.hand_size,  # play card at slot 0
        "agent_1": env.num_moves - 1,  # noop for the non-acting player
    }


def legacy_reset_deck(env, key):
    colors = jnp.arange(env.num_colors)
    ranks = jnp.arange(env.num_ranks)
    ranks = jnp.repeat(ranks, env.num_cards_of_rank)
    color_rank_pairs = jnp.dstack(jnp.meshgrid(colors, ranks)).reshape(-1, 2)
    _, deck_key = jax.random.split(key)
    shuffled_pairs = jax.random.permutation(deck_key, color_rank_pairs, axis=0)
    deck = env._one_hot_encode_deck(shuffled_pairs)
    return deck.at[: env.num_agents * env.hand_size].set(
        jnp.zeros((env.num_colors, env.num_ranks))
    )


# Seat orders used by the routing tests. Seat 0 is the acting seat at reset, so
# every entry puts some agent other than agent_0 there: a permutation that left
# that seat alone would satisfy the routing assertions even with the seat
# mapping removed entirely.
SHUFFLED_SEAT_ORDERS = {
    2: [1, 0],
    3: [1, 2, 0],
    4: [3, 2, 0, 1],
    5: [3, 4, 2, 0, 1],
}


def states_equal(a, b):
    """Whether two states agree on every array field."""
    return all(
        jnp.array_equal(x, y)
        for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b))
    )


def seated(state, seat_order):
    """Put agents in an explicit seat order.

    seat_order only maps seats to agent keys, and cards are dealt to seats
    independently of it, so overriding it yields exactly the state a shuffled
    reset would produce -- without depending on what the RNG happens to permute.
    """
    return state.replace(seat_order=jnp.array(seat_order))


def test_fixed_player_order_preserves_legacy_deck_mapping():
    env = HanabiEnv(shuffle_player_order=False)
    key = jax.random.PRNGKey(42)

    _, state = env.reset(key)

    assert jnp.array_equal(state.seat_order, jnp.arange(env.num_agents))
    assert jnp.array_equal(state.deck, legacy_reset_deck(env, key))


@pytest.mark.parametrize("num_agents", [2, 3, 4, 5])
def test_shuffled_player_order_is_deterministic_and_keeps_agent_keys_stable(num_agents):
    """The shuffled reset yields a valid permutation, reproducibly.

    Which permutation a given key produces is left to the RNG; the routing
    tests pin the orders they need explicitly instead of asserting on it.
    """
    env = HanabiEnv(num_agents=num_agents, shuffle_player_order=True)
    key = jax.random.PRNGKey(42)

    obs, state = env.reset(key)
    obs_again, state_again = env.reset(key)
    legal_moves = env.get_legal_moves(state)

    assert env.agents == [f"agent_{i}" for i in range(num_agents)]
    assert list(obs.keys()) == env.agents
    assert list(obs_again.keys()) == env.agents
    assert list(legal_moves.keys()) == env.agents
    assert list(env.action_spaces.keys()) == env.agents
    assert jnp.array_equal(state.seat_order, state_again.seat_order)
    assert jnp.array_equal(jnp.sort(state.seat_order), jnp.arange(num_agents))


@pytest.mark.parametrize("num_agents", [2, 3, 4, 5])
def test_shuffling_only_reassigns_seats_and_leaves_the_deal_untouched(num_agents):
    """A shuffled reset differs from a fixed one only in seat_order.

    This is what lets the routing tests below build a shuffled state by
    overriding seat_order rather than fishing for a seed.
    """
    key = jax.random.PRNGKey(42)
    _, fixed = HanabiEnv(num_agents=num_agents).reset(key)
    _, shuffled = HanabiEnv(num_agents=num_agents, shuffle_player_order=True).reset(key)

    assert jnp.array_equal(shuffled.deck, fixed.deck)
    assert jnp.array_equal(shuffled.player_hands, fixed.player_hands)
    assert jnp.array_equal(shuffled.cur_player_idx, fixed.cur_player_idx)
    assert jnp.array_equal(
        seated(fixed, shuffled.seat_order).seat_order, shuffled.seat_order
    )


def test_injected_deck_reset_keeps_fixed_seat_order_and_dealt_cards():
    env = HanabiEnv()
    deck = get_decks()[0]

    _, state = env.reset_from_deck(deck)

    assert jnp.array_equal(state.seat_order, jnp.arange(env.num_agents))
    assert jnp.array_equal(state.player_hands[0], deck[: env.hand_size])


def test_shuffled_player_order_routes_current_seat_action_to_assigned_agent_key():
    env = HanabiEnv(num_agents=2)
    # inject the deal rather than drawing one: the assertion below turns on the
    # first card being unplayable, and which card a key deals depends on the JAX
    # PRNG implementation (see the test_obs_function fix in tests/smax)
    _, state = env.reset_from_deck_of_pairs(deck_with_first_card(0, 1))  # R2
    state = seated(state, SHUFFLED_SEAT_ORDERS[2])  # agent_1 holds the acting seat

    assert int(jnp.argmax(state.cur_player_idx)) == 0

    _, next_state, _, _, _ = env.step_env(
        jax.random.PRNGKey(0),
        state,
        {
            "agent_0": 0,  # discard slot 0, wrong player for current seat
            "agent_1": env.hand_size,  # play slot 0, assigned to current seat
        },
    )

    assert int(jnp.sum(next_state.life_tokens)) == env.max_life_tokens - 1


def test_shuffled_player_order_routes_legal_moves_to_assigned_agent_key():
    env = HanabiEnv(num_agents=2)
    _, state = env.reset(jax.random.PRNGKey(42))
    state = seated(state, SHUFFLED_SEAT_ORDERS[2])  # agent_1 holds the acting seat

    legal_moves = env.get_legal_moves(state)

    assert bool(legal_moves["agent_1"][env.play_action_range].any())
    assert not bool(legal_moves["agent_1"][-1])
    assert not bool(legal_moves["agent_0"][env.play_action_range].any())
    assert bool(legal_moves["agent_0"][-1])


def test_shuffled_player_order_routes_observations_to_assigned_agent_key():
    env = HanabiEnv(num_agents=2)
    physical_seat_obs, state = env.reset(jax.random.PRNGKey(42))

    shuffled = seated(state, SHUFFLED_SEAT_ORDERS[2])  # agent_1 holds the acting seat
    obs = env.get_obs(shuffled, shuffled, action=env.num_moves - 1)

    assert jnp.array_equal(obs["agent_1"], physical_seat_obs["agent_0"])
    assert jnp.array_equal(obs["agent_0"], physical_seat_obs["agent_1"])


def test_step_game_terminates_immediately_when_last_life_is_lost():
    env = make_single_life_env()
    state = env.reset_game_from_deck_of_pairs(single_life_loss_deck())

    next_state, _ = env.step_game(state, aidx=0, action=env.hand_size)

    assert bool(next_state.out_of_lives)
    assert bool(next_state.terminal)
    assert int(jnp.sum(next_state.life_tokens)) == 0


def test_step_env_reports_done_on_losing_final_life():
    env = make_single_life_env()
    _, state = env.reset_from_deck_of_pairs(single_life_loss_deck())

    _, next_state, rewards, dones, info = env.step_env(
        jax.random.PRNGKey(0), state, single_life_actions(env)
    )

    assert bool(next_state.terminal)
    assert bool(dones["agent_0"])
    assert bool(dones["agent_1"])
    assert bool(dones["__all__"])
    assert rewards["__all__"] == rewards["agent_0"] == rewards["agent_1"]
    assert info == {}


def test_step_autoresets_after_final_life_loss():
    env = make_single_life_env()
    _, state = env.reset_from_deck_of_pairs(single_life_loss_deck())

    _, next_state, _, dones, _ = env.step(
        jax.random.PRNGKey(0), state, single_life_actions(env)
    )

    assert bool(dones["__all__"])
    assert not bool(next_state.terminal)
    assert not bool(next_state.out_of_lives)
    assert int(jnp.sum(next_state.life_tokens)) == env.max_life_tokens


def test_injected_decks():
    """
    This tests consists in injecting in the Hanabi environment a set of decks and actions that are known to produce a certain score.
    The test checks if the scores produced by the environment are the same as the expected ones.
    """
    print("Hanabi Test: test_injected_decks")
    actions_seq = get_action_sequences()
    decks = get_decks()
    true_scores = get_scores()
    scores = jax.jit(jax.vmap(get_injected_score))(decks, actions_seq)
    assert (true_scores == scores).all(), (
        "The injected decks-actions didn't produce the expeceted scores"
    )
    print("Test passed")


# ---------------------------------------------------------------------------
# New fixtures
# ---------------------------------------------------------------------------


def all_r1_deck():
    """50-card deck where every card is R1 (color 0, rank 0)."""
    return jnp.zeros((env.deck_size, 2), dtype=int)


def deck_with_first_card(color: int, rank: int):
    """50-card deck with (color, rank) as card 0; remainder are R1."""
    deck = np.zeros((env.deck_size, 2), dtype=int)
    deck[0] = [color, rank]
    return jnp.array(deck)


# ---------------------------------------------------------------------------
# Observation shape
# ---------------------------------------------------------------------------


def test_obs_shape_after_reset():
    """Each agent's observation has exactly obs_size features after reset."""
    key = jax.random.PRNGKey(0)
    obs, _ = env.reset(key)
    for agent in env.agents:
        assert obs[agent].shape == (env.obs_size,)


# ---------------------------------------------------------------------------
# Fireworks and scoring
# ---------------------------------------------------------------------------


def test_correct_play_advances_fireworks_and_score():
    """Playing a valid card increments fireworks and score by 1."""
    state = env.reset_game_from_deck_of_pairs(deck_with_first_card(0, 0))  # R1
    next_state, reward = env.step_game(state, aidx=0, action=env.hand_size)
    assert int(next_state.fireworks[0].sum()) == 1
    assert int(next_state.score) == 1
    assert int(reward) == 1


def test_wrong_play_loses_a_life():
    """Playing an invalid card spends a life token without advancing fireworks."""
    state = env.reset_game_from_deck_of_pairs(
        deck_with_first_card(0, 1)
    )  # R2 on empty fireworks
    next_state, reward = env.step_game(state, aidx=0, action=env.hand_size)
    assert int(next_state.life_tokens.sum()) == env.max_life_tokens - 1
    assert int(next_state.fireworks.sum()) == 0
    assert int(reward) == 0


def test_perfect_score_terminates_game():
    """Playing the 25th card triggers terminal and fills all fireworks."""
    state = env.reset_game_from_deck_of_pairs(deck_with_first_card(4, 4))  # B5
    near_perfect = jnp.ones((env.num_colors, env.num_ranks)).at[4, 4].set(0)
    state = state.replace(fireworks=near_perfect)
    next_state, _ = env.step_game(state, aidx=0, action=env.hand_size)
    assert bool(next_state.terminal)
    assert int(next_state.fireworks.sum()) == env.num_colors * env.num_ranks


# ---------------------------------------------------------------------------
# Info token mechanics
# ---------------------------------------------------------------------------


def test_hint_spends_info_token():
    """Giving a hint reduces the info token count by one."""
    state = env.reset_game_from_deck_of_pairs(all_r1_deck())
    initial_tokens = int(state.info_tokens.sum())
    hint_action = 2 * env.hand_size  # first color-hint action (hint R to player 1)
    next_state, _ = env.step_game(state, aidx=0, action=hint_action)
    assert int(next_state.info_tokens.sum()) == initial_tokens - 1


def test_discard_gains_info_token():
    """Discarding when tokens are not full restores one token."""
    state = env.reset_game_from_deck_of_pairs(all_r1_deck())
    tokens = state.info_tokens.at[env.max_info_tokens - 1].set(0)
    state = state.replace(info_tokens=tokens)
    assert int(state.info_tokens.sum()) == env.max_info_tokens - 1
    next_state, _ = env.step_game(state, aidx=0, action=0)  # discard card 0
    assert int(next_state.info_tokens.sum()) == env.max_info_tokens


# ---------------------------------------------------------------------------
# Legal moves
# ---------------------------------------------------------------------------


def test_discard_is_illegal_when_info_tokens_full():
    """No discard action is legal for the acting player at the start of a game."""
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)
    legal = env.get_legal_moves(state)
    acting_seat = int(jnp.nonzero(state.cur_player_idx, size=1)[0][0])
    acting_player = env.agents[int(state.seat_order[acting_seat])]
    assert not bool(legal[acting_player][env.discard_action_range].any())


def test_noop_is_legal_only_for_non_acting_player():
    """Noop is legal exactly for agents who are not the current player."""
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)
    legal = env.get_legal_moves(state)
    noop = env.num_moves - 1
    acting_seat = int(jnp.nonzero(state.cur_player_idx, size=1)[0][0])
    acting_idx = int(state.seat_order[acting_seat])
    for i, agent in enumerate(env.agents):
        if i == acting_idx:
            assert not bool(legal[agent][noop])
        else:
            assert bool(legal[agent][noop])


# ---------------------------------------------------------------------------
# Last-round countdown
# ---------------------------------------------------------------------------


def test_last_round_count_increments_when_deck_empty():
    """last_round_count increases by 1 per turn once the deck is exhausted."""
    state = env.reset_game_from_deck_of_pairs(all_r1_deck())
    state = state.replace(num_cards_dealt=env.deck_size)
    hint_action = 2 * env.hand_size
    next_state, _ = env.step_game(state, aidx=0, action=hint_action)
    assert int(next_state.last_round_count) == 1
    assert not bool(next_state.terminal)


def test_last_round_terminates_game():
    """Game ends when last_round_count reaches num_agents + 1."""
    state = env.reset_game_from_deck_of_pairs(all_r1_deck())
    state = state.replace(
        num_cards_dealt=env.deck_size,
        last_round_count=env.num_agents,  # one step below the terminal threshold
    )
    hint_action = 2 * env.hand_size
    next_state, _ = env.step_game(state, aidx=0, action=hint_action)
    assert bool(next_state.terminal)
    assert int(next_state.last_round_count) == env.num_agents + 1


# ---------------------------------------------------------------------------
# Multi-player configurations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shuffled", [False, True])
@pytest.mark.parametrize("num_agents", [2, 3, 4, 5])
def test_multi_player_reset_and_step(num_agents, shuffled):
    """reset and step work for all valid player counts; obs shapes are correct.

    Run with seats in order and permuted: under a non-identity seat order,
    get_legal_moves and step_env must still agree on which agent key is acting.
    """
    env_mp = HanabiEnv(num_agents=num_agents)
    key = jax.random.PRNGKey(0)
    obs, state = env_mp.reset(key)

    if shuffled:
        state = seated(state, SHUFFLED_SEAT_ORDERS[num_agents])
        obs = env_mp.get_obs(state, state, action=env_mp.num_moves - 1)

    assert len(obs) == num_agents
    for agent in env_mp.agents:
        assert obs[agent].shape == (env_mp.obs_size,)

    assert jnp.array_equal(jnp.sort(state.seat_order), jnp.arange(num_agents))

    acting_seat = int(jnp.nonzero(state.cur_player_idx, size=1)[0][0])
    acting_idx = int(state.seat_order[acting_seat])
    acting_agent = env_mp.agents[acting_idx]

    # the table must move the acting seat, or the assertions below would hold
    # even if seat mapping were dropped entirely
    assert (acting_idx != acting_seat) == shuffled

    # legal moves must be keyed by agent, not seat: exactly the acting agent is
    # barred from noop, and every other agent is restricted to it
    noop = env_mp.num_moves - 1
    legal = env_mp.get_legal_moves(state)
    for agent in env_mp.agents:
        assert bool(legal[agent][noop]) == (agent != acting_agent)

    actions = {agent: noop for agent in env_mp.agents}
    actions[acting_agent] = int(jnp.argmax(legal[acting_agent]))  # first legal move

    obs2, state2, rewards, dones, _ = env_mp.step(key, state, actions)

    assert len(obs2) == num_agents
    assert "__all__" in dones
    assert "__all__" in rewards

    # play passed to the next seat
    assert int(jnp.argmax(state2.cur_player_idx)) == (acting_seat + 1) % num_agents

    # ...and the move that got executed was the acting agent's, not some other
    # agent's noop. The turn advances either way, so compare against the state
    # reached when every agent noops: routing the action by seat instead of by
    # agent would pick up agent_{acting_seat}'s noop and land here instead.
    _, all_noop_state, _, _, _ = env_mp.step(
        key, state, {agent: noop for agent in env_mp.agents}
    )
    assert not states_equal(state2, all_noop_state)


# ---------------------------------------------------------------------------
# Tests for asymmetric num_colors != num_ranks (regression tests for the
# repeat-axis bug in _hint_fn that caused crashes on non-square configs).
# ---------------------------------------------------------------------------


def make_asymmetric_env(num_colors=3, num_ranks=5):
    """Create a HanabiEnv where num_colors != num_ranks."""
    # num_cards_of_rank must have exactly num_ranks entries
    default_cards = np.array([3, 2, 2, 2, 1])
    if num_ranks <= len(default_cards):
        num_cards_of_rank = default_cards[:num_ranks]
    else:
        num_cards_of_rank = np.concatenate(
            [default_cards, np.full(num_ranks - len(default_cards), 2)]
        )
    return HanabiEnv(
        num_agents=2,
        num_colors=num_colors,
        num_ranks=num_ranks,
        hand_size=5,
        max_info_tokens=8,
        max_life_tokens=3,
        num_cards_of_rank=num_cards_of_rank,
    )


def test_asymmetric_reset_and_step_smoke():
    """Reset and take every action type with num_colors != num_ranks without crashing."""
    for num_colors, num_ranks in [(3, 5), (5, 3)]:
        env = make_asymmetric_env(num_colors=num_colors, num_ranks=num_ranks)
        key = jax.random.PRNGKey(42)
        obs, state = env.reset(key)

        # verify observation shapes are correct
        for agent in env.agents:
            assert obs[agent].shape == (env.obs_size,), (
                f"obs shape mismatch for {num_colors}c/{num_ranks}r: "
                f"got {obs[agent].shape}, expected ({env.obs_size},)"
            )

        # take one action of each type: discard, play, color hint, rank hint
        actions_to_try = [
            0,  # discard card 0
            env.hand_size,  # play card 0
            env.color_action_range[0],  # hint color to other player
            env.rank_action_range[0],  # hint rank to other player
        ]
        for action in actions_to_try:
            key, subkey = jax.random.split(key)
            cur_player = jnp.nonzero(state.cur_player_idx, size=1)[0][0]
            actions = {agent: env.num_moves - 1 for agent in env.agents}  # noop
            actions[env.agents[int(cur_player)]] = int(action)
            obs, state, rewards, dones, info = env.step(subkey, state, actions)

    print("test_asymmetric_reset_and_step_smoke passed")


def test_asymmetric_color_hint_knowledge():
    """After a color hint with num_colors != num_ranks, card_knowledge must have
    the correct shape and eliminate the right possibilities."""
    env = make_asymmetric_env(num_colors=3, num_ranks=5)
    key = jax.random.PRNGKey(0)
    state = env.reset_game(key)

    # player 0 gives a color-0 hint to player 1
    color_hint_action = int(env.color_action_range[0])  # hint color 0 to next player
    new_state, reward = env.step_game(state, aidx=0, action=color_hint_action)

    # card_knowledge for player 1 should still have shape (hand_size, num_colors * num_ranks)
    p1_knowledge = new_state.card_knowledge[1]
    assert p1_knowledge.shape == (env.hand_size, env.num_colors * env.num_ranks), (
        f"knowledge shape wrong: {p1_knowledge.shape}"
    )

    # for cards that DON'T match color 0, the color-0 columns should be zeroed out
    p1_hand = new_state.player_hands[1]
    card_colors = jnp.sum(p1_hand, axis=2)  # (hand_size, num_colors)
    hint_color_vec = jnp.zeros(env.num_colors).at[0].set(1)  # color 0
    matches = jnp.matmul(card_colors, hint_color_vec)  # which cards have color 0

    knowledge_reshaped = p1_knowledge.reshape(
        env.hand_size, env.num_colors, env.num_ranks
    )
    for card_idx in range(env.hand_size):
        if not p1_hand[card_idx].any():
            continue  # skip empty card slots
        if matches[card_idx] == 0:
            # card does NOT have color 0 → color-0 row should be all zeros
            assert jnp.all(knowledge_reshaped[card_idx, 0, :] == 0), (
                f"card {card_idx} doesn't match color 0 but knowledge wasn't zeroed"
            )
        else:
            # card HAS color 0 → other color rows should be all zeros
            for c in range(1, env.num_colors):
                assert jnp.all(knowledge_reshaped[card_idx, c, :] == 0), (
                    f"card {card_idx} matches color 0 but color {c} wasn't zeroed"
                )

    print("test_asymmetric_color_hint_knowledge passed")


def test_asymmetric_rank_hint_knowledge():
    """After a rank hint with num_colors != num_ranks, card_knowledge must have
    the correct shape and eliminate the right possibilities."""
    env = make_asymmetric_env(num_colors=3, num_ranks=5)
    key = jax.random.PRNGKey(0)
    state = env.reset_game(key)

    # player 0 gives a rank-0 hint to player 1
    rank_hint_action = int(env.rank_action_range[0])  # hint rank 0 to next player
    new_state, reward = env.step_game(state, aidx=0, action=rank_hint_action)

    p1_knowledge = new_state.card_knowledge[1]
    assert p1_knowledge.shape == (env.hand_size, env.num_colors * env.num_ranks)

    p1_hand = new_state.player_hands[1]
    card_ranks = jnp.sum(p1_hand, axis=1)  # (hand_size, num_ranks)
    hint_rank_vec = jnp.zeros(env.num_ranks).at[0].set(1)
    matches = jnp.matmul(card_ranks, hint_rank_vec)

    knowledge_reshaped = p1_knowledge.reshape(
        env.hand_size, env.num_colors, env.num_ranks
    )
    for card_idx in range(env.hand_size):
        if not p1_hand[card_idx].any():
            continue
        if matches[card_idx] == 0:
            # card does NOT have rank 0 → rank-0 column should be all zeros
            assert jnp.all(knowledge_reshaped[card_idx, :, 0] == 0), (
                f"card {card_idx} doesn't match rank 0 but knowledge wasn't zeroed"
            )
        else:
            # card HAS rank 0 → other rank columns should be all zeros
            for r in range(1, env.num_ranks):
                assert jnp.all(knowledge_reshaped[card_idx, :, r] == 0), (
                    f"card {card_idx} matches rank 0 but rank {r} wasn't zeroed"
                )

    print("test_asymmetric_rank_hint_knowledge passed")


def test_asymmetric_full_game_rollout():
    """Run a complete game with num_colors != num_ranks using random legal actions."""
    for num_colors, num_ranks in [(3, 5), (5, 3), (2, 4)]:
        env = make_asymmetric_env(num_colors=num_colors, num_ranks=num_ranks)
        key = jax.random.PRNGKey(99)
        obs, state = env.reset(key)

        max_steps = env.deck_size + env.num_agents + 10  # generous upper bound
        for step_i in range(max_steps):
            if state.terminal:
                break

            key, subkey, action_key = jax.random.split(key, 3)
            cur_player = int(jnp.nonzero(state.cur_player_idx, size=1)[0][0])
            legal_moves = env.get_legal_moves(state)
            cur_legal = legal_moves[env.agents[cur_player]]

            # pick a random legal action
            legal_indices = jnp.where(cur_legal, size=env.num_moves)[0]
            num_legal = int(cur_legal.sum())
            action_idx = jax.random.randint(action_key, (), 0, max(num_legal, 1))
            action = int(legal_indices[action_idx])

            actions = {agent: env.num_moves - 1 for agent in env.agents}
            actions[env.agents[cur_player]] = action
            obs, state, rewards, dones, info = env.step(subkey, state, actions)

        assert step_i > 0, f"Game with {num_colors}c/{num_ranks}r ended immediately"

    print("test_asymmetric_full_game_rollout passed")


# ---------------------------------------------------------------------------
# Six colors: the multicolor ("M") suit played as an ordinary sixth color.
# ---------------------------------------------------------------------------

SIXTH_COLOR = 5  # index of the multicolor suit


def make_six_color_env(num_agents=2):
    """Standard Hanabi plus the sixth (multicolor) suit."""
    return HanabiEnv(num_agents=num_agents, num_colors=6)


def six_color_deck(*cards):
    """Deck of (color, rank) pairs; the given cards are dealt first, rest are R1."""
    deck = np.zeros((60, 2), dtype=int)
    for i, (color, rank) in enumerate(cards):
        deck[i] = [color, rank]
    return jnp.array(deck)


def test_six_color_default_color_map():
    """The sixth color is labelled M; five-color envs are unchanged."""
    assert make_six_color_env().color_map == ["R", "Y", "G", "W", "B", "M"]
    assert HanabiEnv().color_map == ["R", "Y", "G", "W", "B"]


def test_six_color_env_dimensions():
    """Deck, action and observation sizes all scale with the sixth color."""
    env_6 = make_six_color_env()

    assert env_6.num_colors == 6
    assert env_6.deck_size == 60  # 6 suits * 10 cards
    # 5 discards + 5 plays + (6 colors + 5 ranks) hints for 1 teammate + noop
    assert env_6.num_moves == 22

    expected_obs_size = sum(
        [
            1 * 5 * 6 * 5 + 2,  # hands: teammate's cards + missing-card flags
            (60 - 2 * 5) + 6 * 5 + 8 + 3,  # board: deck, fireworks, info, lives
            6 * 10,  # discards
            2 + 4 + 2 + 6 + 5 + 5 + 5 + 6 * 5 + 1 + 1,  # last action
            2 * 5 * (6 * 5 + 6 + 5),  # v0 belief
        ]
    )
    assert env_6.obs_size == expected_obs_size == 774

    obs, state = env_6.reset(jax.random.PRNGKey(0))
    for agent in env_6.agents:
        assert obs[agent].shape == (env_6.obs_size,)
    assert state.fireworks.shape == (6, 5)
    assert state.player_hands.shape == (2, 5, 6, 5)


def test_six_color_deck_composition():
    """Every one of the 6 suits contributes 3/2/2/2/1 cards of ranks 1-5."""
    env_6 = make_six_color_env()
    counts = np.array(env_6.get_full_deck().sum(axis=0))
    expected = np.tile(env_6.num_cards_of_rank, (6, 1))
    assert counts.shape == (6, 5)
    assert (counts == expected).all()

    # a shuffled deck holds the same multiset of cards
    state = env_6.reset_game(jax.random.PRNGKey(1))
    dealt = state.player_hands.sum(axis=(0, 1))
    assert (np.array(state.deck.sum(axis=0) + dealt) == expected).all()


def test_six_color_action_encoding_covers_sixth_color():
    """A hint action exists for M, and there is exactly one per color per target."""
    env_6 = make_six_color_env()
    color_hints = [
        env_6.action_encoding[int(a)] for a in env_6.color_action_range.tolist()
    ]
    assert color_hints == [f"H{c} to P1 relative" for c in env_6.color_map]
    assert len(env_6.color_action_range) == 6


def test_six_color_hint_updates_knowledge_for_sixth_color():
    """Hinting M splits the target's hand into M and definitely-not-M cards."""
    env_6 = make_six_color_env()
    # agent 0 holds cards 0-4, agent 1 holds cards 5-9: two M1s then three R1s
    deck = six_color_deck(
        *[(0, 0)] * 5,
        (SIXTH_COLOR, 0),
        (SIXTH_COLOR, 0),
        (0, 0),
        (0, 0),
        (0, 0),
    )
    state = env_6.reset_game_from_deck_of_pairs(deck)

    hint_m = int(env_6.color_action_range[SIXTH_COLOR])  # hint M to the next player
    assert env_6.action_encoding[hint_m] == "HM to P1 relative"
    new_state, _ = env_6.step_game(state, aidx=0, action=hint_m)

    knowledge = new_state.card_knowledge[1].reshape(env_6.hand_size, 6, 5)
    assert bool(new_state.colors_revealed[1][:, SIXTH_COLOR].sum() == 2)
    for card_idx in range(env_6.hand_size):
        is_m = card_idx < 2
        if is_m:
            # only the M row survives
            assert jnp.all(knowledge[card_idx, :SIXTH_COLOR, :] == 0)
            assert jnp.any(knowledge[card_idx, SIXTH_COLOR, :] > 0)
        else:
            # M is ruled out, other colors remain
            assert jnp.all(knowledge[card_idx, SIXTH_COLOR, :] == 0)
            assert jnp.any(knowledge[card_idx, :SIXTH_COLOR, :] > 0)


def test_six_color_play_starts_sixth_firework():
    """Playing M1 onto an empty board scores and starts the M firework."""
    env_6 = make_six_color_env()
    state = env_6.reset_game_from_deck_of_pairs(six_color_deck((SIXTH_COLOR, 0)))

    next_state, reward = env_6.step_game(state, aidx=0, action=env_6.hand_size)

    assert int(reward) == 1
    assert int(next_state.score) == 1
    assert int(next_state.fireworks[SIXTH_COLOR].sum()) == 1
    assert int(next_state.fireworks.sum()) == 1


def test_six_color_perfect_score_is_thirty():
    """The sixth suit raises the maximum score from 25 to 30."""
    env_6 = make_six_color_env()
    state = env_6.reset_game_from_deck_of_pairs(six_color_deck((SIXTH_COLOR, 4)))
    near_perfect = jnp.ones((6, 5)).at[SIXTH_COLOR, 4].set(0)
    state = state.replace(fireworks=near_perfect)

    next_state, reward = env_6.step_game(state, aidx=0, action=env_6.hand_size)

    assert bool(next_state.terminal)
    assert int(next_state.fireworks.sum()) == 30 == env_6.num_colors * env_6.num_ranks
    assert int(reward) == 1


@pytest.mark.parametrize("num_agents", [2, 3, 4, 5])
def test_six_color_multi_player_reset_and_step(num_agents):
    """The sixth color works for every supported player count."""
    env_6 = make_six_color_env(num_agents=num_agents)
    key = jax.random.PRNGKey(0)
    obs, state = env_6.reset(key)

    assert len(obs) == num_agents
    for agent in env_6.agents:
        assert obs[agent].shape == (env_6.obs_size,)

    acting_idx = int(jnp.nonzero(state.cur_player_idx, size=1)[0][0])
    actions = {agent: env_6.num_moves - 1 for agent in env_6.agents}
    actions[env_6.agents[acting_idx]] = int(env_6.color_action_range[SIXTH_COLOR])

    obs2, _, rewards, dones, _ = env_6.step(key, state, actions)

    assert len(obs2) == num_agents
    assert "__all__" in dones and "__all__" in rewards


def test_six_color_full_game_rollout():
    """Play a whole six-color game with random legal actions."""
    env_6 = make_six_color_env()
    key = jax.random.PRNGKey(7)
    obs, state = env_6.reset(key)

    steps = 0
    while not bool(state.terminal) and steps < env_6.deck_size + 20:
        key, subkey, action_key = jax.random.split(key, 3)
        cur_player = int(jnp.nonzero(state.cur_player_idx, size=1)[0][0])
        cur_legal = env_6.get_legal_moves(state)[env_6.agents[cur_player]]
        legal_indices = jnp.where(cur_legal, size=env_6.num_moves)[0]
        num_legal = int(cur_legal.sum())
        action = int(legal_indices[jax.random.randint(action_key, (), 0, num_legal)])

        actions = {agent: env_6.num_moves - 1 for agent in env_6.agents}
        actions[env_6.agents[cur_player]] = action
        obs, state, rewards, dones, info = env_6.step_env(subkey, state, actions)
        steps += 1

    assert bool(state.terminal), "six-color game did not terminate"
    assert 0 <= int(state.score) <= env_6.num_colors * env_6.num_ranks
    for agent in env_6.agents:
        assert obs[agent].shape == (env_6.obs_size,)


# ---------------------------------------------------------------------------
# color_map configuration
# ---------------------------------------------------------------------------


def test_custom_color_map_is_used_for_labels():
    """An explicit color_map drives hint labels and card rendering."""
    env_c = HanabiEnv(num_agents=2, num_colors=6, color_map=list("123456"))
    assert env_c.action_encoding[int(env_c.color_action_range[SIXTH_COLOR])] == (
        "H6 to P1 relative"
    )
    card = jnp.zeros((6, 5)).at[SIXTH_COLOR, 0].set(1)
    assert env_c.card_to_string(card) == "61"


def test_color_map_length_must_match_num_colors():
    """Mismatched or missing color labels are rejected at construction."""
    with pytest.raises(AssertionError):
        HanabiEnv(num_agents=2, num_colors=6, color_map=["R", "Y", "G", "W", "B"])
    with pytest.raises(AssertionError):
        # no default labels beyond the six known suits
        HanabiEnv(num_agents=2, num_colors=7)


def test_num_cards_of_rank_length_must_match_num_ranks():
    """A rank-count array of the wrong length is rejected at construction."""
    with pytest.raises(AssertionError):
        HanabiEnv(num_agents=2, num_colors=6, num_ranks=5, num_cards_of_rank=[3, 2, 1])


def main():
    test_injected_decks()
    test_asymmetric_reset_and_step_smoke()
    test_asymmetric_color_hint_knowledge()
    test_asymmetric_rank_hint_knowledge()
    test_asymmetric_full_game_rollout()
    test_six_color_env_dimensions()
    test_six_color_deck_composition()
    test_six_color_hint_updates_knowledge_for_sixth_color()
    test_six_color_full_game_rollout()


if __name__ == "__main__":
    main()

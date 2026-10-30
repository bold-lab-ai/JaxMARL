import jax.numpy as jnp
import pytest


def compute_qmix_target(rewards, dones, qmix_next, gamma):
    """Computes QMIX TD targets as in baselines/QLearning/qmix_rnn.py."""
    return (
        rewards[:-1]
        + (1 - dones[1:])  # corrected slice
        * gamma
        * qmix_next[1:]
    )


def compute_vdn_target(rewards, dones, q_next_sum, gamma):
    """Computes VDN TD targets as in baselines/QLearning/vdn_rnn.py."""
    return (
        rewards[:-1]
        + (1 - dones[1:])  # corrected slice
        * gamma
        * q_next_sum[1:]
    )


def compute_iql_target(rewards, dones, q_next, gamma):
    """Computes IQL TD targets as in baselines/QLearning/iql_rnn.py."""
    return (
        rewards[:, :-1]
        + (1 - dones[:, 1:])  # corrected slice
        * gamma
        * q_next[:, 1:]
    )


def test_qmix_target_episode_boundary_mask():
    """Verify TD target correctly masks bootstrapping at episode boundaries.

    In recurrent baselines (qmix_rnn, vdn_rnn, iql_rnn), Timestep.dones stores
    last_dones (the done signal from the PREVIOUS transition, used for RNN carry reset).
    Therefore:
      - dones[0] is initial done (False)
      - dones[t+1] is the done resulting from taking action at step t (reward[t])
      - To align with rewards[:-1] and q_next[1:], dones must be sliced as dones[1:]
    """
    gamma = 0.99
    # Suppose trajectory has 5 timesteps (t = 0, 1, 2, 3, 4)
    # Transition 1 terminates early (e.g. SMAX battle ends)
    # Step 0: non-terminal (d_0 = False)
    # Step 1: terminal (d_1 = True) -> episode ends
    # Step 2: first step of new episode (d_2 = False)
    # Step 3: non-terminal (d_3 = False)
    # Step 4: terminal (d_4 = True)
    #
    # Because Timestep.dones stores last_dones:
    # dones[0] = False (d_{-1})
    # dones[1] = False (d_0)
    # dones[2] = True  (d_1)
    # dones[3] = False (d_2)
    # dones[4] = False (d_3)
    dones = jnp.array([False, False, True, False, False], dtype=jnp.float32)
    rewards = jnp.array([1.0, 2.0, 0.5, 1.5, 3.0], dtype=jnp.float32)
    q_next = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=jnp.float32)

    # Correct targets for transitions t = 0, 1, 2, 3:
    # t=0: r_0 + gamma * (1 - d_0) * Q_1 = 1.0 + 0.99 * (1 - 0) * 20.0 = 20.8
    # t=1: r_1 + gamma * (1 - d_1) * Q_2 = 2.0 + 0.99 * (1 - 1) * 30.0 = 2.0  (no bootstrapping!)
    # t=2: r_2 + gamma * (1 - d_2) * Q_3 = 0.5 + 0.99 * (1 - 0) * 40.0 = 40.1
    # t=3: r_3 + gamma * (1 - d_3) * Q_4 = 1.5 + 0.99 * (1 - 0) * 50.0 = 51.0
    expected_targets = jnp.array(
        [
            1.0 + gamma * 20.0,
            2.0,  # masked out because transition 1 is terminal!
            0.5 + gamma * 40.0,
            1.5 + gamma * 50.0,
        ]
    )

    targets = compute_qmix_target(rewards, dones, q_next, gamma)
    assert jnp.allclose(targets, expected_targets)

    # Verify that the old buggy slice (dones[:-1]) produces the WRONG values:
    buggy_targets = rewards[:-1] + (1 - dones[:-1]) * gamma * q_next[1:]
    # Bug 1: transition 1 was terminal, but buggy code bootstraps anyway (leaking Q value!)
    assert not jnp.isclose(buggy_targets[1], expected_targets[1])
    # Bug 2: transition 2 was NOT terminal, but buggy code zeroes out the bootstrap!
    assert not jnp.isclose(buggy_targets[2], expected_targets[2])


def test_vdn_target_episode_boundary_mask():
    """Verify VDN TD targets correctly mask bootstrapping at episode boundaries."""
    gamma = 0.95
    # dones stored as last_dones:
    # Transition 0: terminal (d_0 = True) -> stored at dones[1]
    dones = jnp.array([False, True, False, False], dtype=jnp.float32)
    rewards = jnp.array([5.0, 1.0, 2.0, 3.0], dtype=jnp.float32)
    q_next_sum = jnp.array([10.0, 15.0, 20.0, 25.0], dtype=jnp.float32)

    # t=0 is terminal -> target is just r_0 = 5.0
    # t=1 is non-terminal -> target is r_1 + gamma * Q_2 = 1.0 + 0.95 * 20.0 = 20.0
    # t=2 is non-terminal -> target is r_2 + gamma * Q_3 = 2.0 + 0.95 * 25.0 = 25.75
    expected_targets = jnp.array(
        [
            5.0,
            1.0 + gamma * 20.0,
            2.0 + gamma * 25.0,
        ]
    )

    targets = compute_vdn_target(rewards, dones, q_next_sum, gamma)
    assert jnp.allclose(targets, expected_targets)


def test_iql_target_episode_boundary_mask():
    """Verify per-agent IQL TD targets correctly mask bootstrapping."""
    gamma = 0.90
    num_agents = 2
    # dones shape: (num_agents, timesteps)
    # Agent 0 terminates at step 1 (dones[0, 2] = 1.0)
    # Agent 1 terminates at step 0 (dones[1, 1] = 1.0)
    dones = jnp.array(
        [
            [False, False, True, False],
            [False, True, False, False],
        ],
        dtype=jnp.float32,
    )

    rewards = jnp.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ],
        dtype=jnp.float32,
    )

    q_next = jnp.array(
        [
            [10.0, 20.0, 30.0, 40.0],
            [10.0, 20.0, 30.0, 40.0],
        ],
        dtype=jnp.float32,
    )

    expected_targets = jnp.array(
        [
            [1.0 + gamma * 20.0, 2.0, 3.0 + gamma * 40.0],
            [5.0, 6.0 + gamma * 30.0, 7.0 + gamma * 40.0],
        ]
    )

    targets = compute_iql_target(rewards, dones, q_next, gamma)
    assert jnp.allclose(targets, expected_targets)

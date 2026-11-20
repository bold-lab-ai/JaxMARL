"""Grid Predator-Prey from IC3Net (Singh, Jain & Sukhbaatar, ICLR 2019).

A JAX port of ``predator_prey_env.py`` from the reference implementation,
https://github.com/IC3Net/IC3Net/tree/master/ic3net-envs. ``num_agents``
predators search a ``dim`` x ``dim`` grid for a single fixed prey. Each predator
sees a ``(2 * vision + 1)`` square window around itself.
"""

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jaxtyping import Array, Bool, Int, PRNGKeyArray

from jaxmarl.environments.multi_agent_env import (
    Actions,
    Dones,
    Infos,
    MultiAgentEnv,
    Observations,
    Rewards,
)
from jaxmarl.environments.multi_agent_env import State as BaseState
from jaxmarl.environments.spaces import Box, Discrete

MODES = ("cooperative", "competitive", "mixed")

# (0: UP, 1: RIGHT, 2: DOWN, 3: LEFT, 4: STAY)
ACTION_DELTAS = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1], [0, 0]])


@struct.dataclass
class State(BaseState):
    """State of the grid Predator-Prey environment.

    Attributes:
        predator_loc: (row, col) of each predator.
        prey_loc: (row, col) of the prey.
        reached_prey: Whether each predator has reached the prey this episode.
    """

    predator_loc: Int[Array, "num_agents 2"]
    prey_loc: Int[Array, "2"]
    reached_prey: Bool[Array, " num_agents"]


class PredatorPreyGrid(MultiAgentEnv):
    """Grid Predator-Prey with cooperative, competitive and mixed rewards.

    Every predator receives ``-0.05`` per step until it reaches the prey. A
    predator on the prey then receives, with ``n`` predators on the prey:
    ``0.05 * n`` (cooperative), ``0.05 / n`` (competitive) or ``0`` (mixed).
    Predators stop moving once they reach the prey. Mixed episodes end when
    every predator has reached the prey; all episodes end after ``max_steps``.

    The paper's configurations are easy (``num_agents=3, dim=5, vision=0,
    max_steps=20``, the defaults), medium (``5, 10, 1, 40``) and hard
    (``10, 20, 1, 80``).

    Args:
        num_agents: Number of predators.
        dim: Side length of the square grid.
        vision: Radius of each predator's square observation window.
        max_steps: Episode length limit.
        mode: Reward structure, one of ``"cooperative"``, ``"competitive"``
            or ``"mixed"``.
        stay: Whether predators have a STAY action. The reference's
            ``--no_stay`` corresponds to ``stay=False``.
    """

    def __init__(
        self,
        num_agents: int = 3,
        dim: int = 5,
        vision: int = 0,
        max_steps: int = 20,
        mode: str = "mixed",
        stay: bool = True,
    ) -> None:
        super().__init__(num_agents)
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if dim * dim < num_agents + 1:
            raise ValueError(
                f"A {dim}x{dim} grid cannot hold {num_agents} predators and a prey"
            )

        self.dim = dim
        self.vision = vision
        self.max_steps = max_steps
        self.mode = mode
        self.stay = stay

        self.TIMESTEP_PENALTY = -0.05
        self.PREY_REWARD = 0.0
        self.POS_PREY_REWARD = 0.05

        # Vocabulary of the one-hot cell encoding, as in the reference: one id
        # per grid cell, then outside, prey and predator classes.
        self.base = dim * dim
        self.outside_class = self.base + 1
        self.prey_class = self.base + 2
        self.predator_class = self.base + 3
        self.vocab_size = self.base + 4
        self.vis_size = 2 * vision + 1

        self.obs_dim = self.vis_size * self.vis_size * self.vocab_size

        self.num_actions = 5 if stay else 4
        # Cells can hold several predators, so counts reach num_agents.
        self.observation_spaces = {
            a: Box(0.0, float(num_agents), (self.obs_dim,)) for a in self.agents
        }
        self.action_spaces = {a: Discrete(self.num_actions) for a in self.agents}

        grid = jnp.arange(self.base, dtype=jnp.int32).reshape(dim, dim)
        grid = jnp.pad(grid, vision, constant_values=self.outside_class)
        self.empty_grid = jax.nn.one_hot(grid, self.vocab_size)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: PRNGKeyArray) -> Tuple[Observations, State]:
        """Place the predators and the prey on distinct random cells."""
        cells = jax.random.choice(
            key, self.base, shape=(self.num_agents + 1,), replace=False
        )
        locs = jnp.stack([cells // self.dim, cells % self.dim], axis=-1)
        state = State(
            predator_loc=locs[: self.num_agents],
            prey_loc=locs[self.num_agents],
            reached_prey=jnp.zeros(self.num_agents, dtype=jnp.bool_),
            done=jnp.array(False),
            step=jnp.array(0),
        )
        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self, key: PRNGKeyArray, state: State, actions: Actions
    ) -> Tuple[Observations, State, Rewards, Dones, Infos]:
        """Move the predators, then score them against the prey."""
        # The baselines pass each action as a scalar or a (1,) array.
        acts = jnp.array([actions[a] for a in self.agents]).reshape(self.num_agents)
        moved = jnp.clip(state.predator_loc + ACTION_DELTAS[acts], 0, self.dim - 1)
        predator_loc = jnp.where(state.reached_prey[:, None], state.predator_loc, moved)

        on_prey = jnp.all(predator_loc == state.prey_loc, axis=-1)
        num_on_prey = jnp.sum(on_prey)
        if self.mode == "cooperative":
            prey_reward = self.POS_PREY_REWARD * num_on_prey
        elif self.mode == "competitive":
            prey_reward = self.POS_PREY_REWARD / jnp.maximum(num_on_prey, 1)
        else:
            prey_reward = self.PREY_REWARD
        reward = jnp.where(on_prey, prey_reward, self.TIMESTEP_PENALTY)

        reached_prey = state.reached_prey | on_prey
        step = state.step + 1
        done = step >= self.max_steps
        if self.mode == "mixed":
            done = done | jnp.all(reached_prey)

        state = state.replace(
            predator_loc=predator_loc,
            reached_prey=reached_prey,
            done=done,
            step=step,
        )
        rewards = {a: reward[i] for i, a in enumerate(self.agents)}
        dones = {a: done for a in self.agents}
        dones["__all__"] = done
        # Success, as in the reference: every predator is on the prey.
        success = num_on_prey == self.num_agents
        infos = {"success": jnp.full((self.num_agents,), success)}
        return self.get_obs(state), state, rewards, dones, infos

    def get_obs(self, state: State) -> Observations:
        """Vision window around each predator, flattened from ``(2v+1, 2v+1, vocab)``.

        Each cell holds a one-hot id of the grid cell (or outside) plus the number
        of predators and prey in it.
        """
        grid = self.empty_grid
        grid = grid.at[
            state.predator_loc[:, 0] + self.vision,
            state.predator_loc[:, 1] + self.vision,
            self.predator_class,
        ].add(1.0)
        grid = grid.at[
            state.prey_loc[0] + self.vision,
            state.prey_loc[1] + self.vision,
            self.prey_class,
        ].add(1.0)

        def _window(loc):
            window = jax.lax.dynamic_slice(
                grid,
                (loc[0], loc[1], 0),
                (self.vis_size, self.vis_size, self.vocab_size),
            )
            return window.reshape(-1)

        obs = jax.vmap(_window)(state.predator_loc)
        return {a: obs[i] for i, a in enumerate(self.agents)}

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: State) -> dict:
        """All actions are always available."""
        return {a: jnp.ones(self.num_actions, dtype=jnp.bool_) for a in self.agents}

    @property
    def agent_classes(self) -> dict:
        return {"predator": self.agents}

    # Matplotlib rendering, compatible with ``jaxmarl.viz.visualizer.Visualizer``.
    _COLORS = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    def init_render(self, ax, state: State):
        """Draw the grid, prey and predators; returns the artists to update."""
        import numpy as np
        from matplotlib.patches import Circle, RegularPolygon

        ax.clear()
        ax.set_xlim(-0.5, self.dim - 0.5)
        ax.set_ylim(-0.5, self.dim - 0.5)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks(range(self.dim))
        ax.set_yticks(range(self.dim))
        ax.grid(True, linewidth=0.5, alpha=0.3)
        ax.set_title(f"Predator-Prey {self.dim}x{self.dim}  step {int(state.step)}")

        prey_r, prey_c = (int(x) for x in state.prey_loc)
        prey = RegularPolygon(
            (prey_c, prey_r),
            numVertices=5,
            radius=0.35,
            color="gold",
            ec="darkgoldenrod",
            lw=1.5,
            zorder=5,
        )
        ax.add_patch(prey)

        locs = np.array(state.predator_loc)
        reached = np.array(state.reached_prey)
        patches, labels = [], []
        for i in range(self.num_agents):
            r, c = int(locs[i, 0]), int(locs[i, 1])
            circle = Circle(
                (c, r),
                0.3,
                color=self._COLORS[i % len(self._COLORS)],
                ec="gold" if reached[i] else "black",
                lw=2.5 if reached[i] else 1.0,
                zorder=10,
            )
            ax.add_patch(circle)
            label = ax.text(
                c,
                r,
                str(i),
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white",
                zorder=11,
            )
            patches.append(circle)
            labels.append(label)
        return {"ax": ax, "prey": prey, "patches": patches, "labels": labels}

    def update_render(self, artists, state: State):
        """Move the artists returned by ``init_render`` to ``state``."""
        import numpy as np

        artists["ax"].set_title(
            f"Predator-Prey {self.dim}x{self.dim}  step {int(state.step)}"
        )
        prey_r, prey_c = (int(x) for x in state.prey_loc)
        artists["prey"].xy = (prey_c, prey_r)
        locs = np.array(state.predator_loc)
        reached = np.array(state.reached_prey)
        for i in range(self.num_agents):
            r, c = int(locs[i, 0]), int(locs[i, 1])
            artists["patches"][i].center = (c, r)
            artists["patches"][i].set_edgecolor("gold" if reached[i] else "black")
            artists["patches"][i].set_linewidth(2.5 if reached[i] else 1.0)
            artists["labels"][i].set_position((c, r))
        return artists

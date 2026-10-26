"""Grid Traffic Junction from CommNet and IC3Net (Singh, Jain & Sukhbaatar, ICLR 2019).

A JAX port of ``traffic_junction_env.py`` and ``traffic_helper.py`` from the
reference implementation, https://github.com/IC3Net/IC3Net/tree/master/ic3net-envs.
Cars enter at the ends of the roads with probability ``add_rate`` per arrival
point and step, follow a fixed route through the junctions, and leave at its
end. Each car chooses GAS (advance one cell) or BRAKE (stay).
"""

from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
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

# Settings of the paper's three versions (Table 4 and the reference README):
# number of cars, grid size, episode length and the final arrival rate of the
# p_arrive curriculum.
DIFFICULTIES = {
    "easy": dict(num_agents=5, dim=6, max_steps=20, add_rate=0.3),
    "medium": dict(num_agents=10, dim=14, max_steps=40, add_rate=0.2),
    "hard": dict(num_agents=20, dim=18, max_steps=80, add_rate=0.05),
}

GAS, BRAKE = 0, 1


# Route construction, ported from the reference ``traffic_helper.py``. It runs
# once in ``__init__`` with numpy.

_MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _road_blocks(w: int, h: int, difficulty: str) -> list:
    """Slices of the grid that are road, one lane per direction."""
    if difficulty == "easy":
        return [np.s_[h // 2, :], np.s_[:, w // 2]]
    if difficulty == "medium":
        return [np.s_[h // 2 - 1 : h // 2 + 1, :], np.s_[:, w // 2 - 1 : w // 2 + 1]]
    return [
        np.s_[h // 3 - 2 : h // 3, :],
        np.s_[2 * h // 3 : 2 * h // 3 + 2, :],
        np.s_[:, w // 3 - 2 : w // 3],
        np.s_[:, 2 * h // 3 : 2 * h // 3 + 2],
    ]


def _junction_layout(dims, grid, difficulty):
    """Arrival and finish points, lane directions and junction cells."""
    h, w = dims
    road_dir = grid.copy()
    junction = np.zeros_like(grid)
    if difficulty == "medium":
        arrival = [(0, w // 2 - 1), (h - 1, w // 2), (h // 2, 0), (h // 2 - 1, w - 1)]
        finish = [(0, w // 2), (h - 1, w // 2 - 1), (h // 2 - 1, 0), (h // 2, w - 1)]
        road_dir[h // 2, :] = 2
        road_dir[h // 2 - 1, :] = 3
        road_dir[:, w // 2] = 4
        junction[h // 2 - 1 : h // 2 + 1, w // 2 - 1 : w // 2 + 1] = 1
    else:
        arrival = [
            (0, w // 3 - 2),
            (0, 2 * w // 3),
            (h // 3 - 1, 0),
            (2 * h // 3 + 1, 0),
            (h - 1, w // 3 - 1),
            (h - 1, 2 * w // 3 + 1),
            (h // 3 - 2, w - 1),
            (2 * h // 3, w - 1),
        ]
        finish = [
            (0, w // 3 - 1),
            (0, 2 * w // 3 + 1),
            (h // 3 - 2, 0),
            (2 * h // 3, 0),
            (h - 1, w // 3 - 2),
            (h - 1, 2 * w // 3),
            (h // 3 - 1, w - 1),
            (2 * h // 3 + 1, w - 1),
        ]
        road_dir[h // 3 - 1, :] = 2
        road_dir[2 * h // 3, :] = 3
        road_dir[2 * h // 3 + 1, :] = 4
        road_dir[:, w // 3 - 2] = 5
        road_dir[:, w // 3 - 1] = 6
        road_dir[:, 2 * w // 3] = 7
        road_dir[:, 2 * w // 3 + 1] = 8
        junction[h // 3 - 2 : h // 3, w // 3 - 2 : w // 3] = 1
        junction[2 * h // 3 : 2 * h // 3 + 2, w // 3 - 2 : w // 3] = 1
        junction[h // 3 - 2 : h // 3, 2 * w // 3 : 2 * w // 3 + 2] = 1
        junction[2 * h // 3 : 2 * h // 3 + 2, 2 * w // 3 : 2 * w // 3 + 2] = 1
    return arrival, finish, road_dir, junction


def _next_move(curr, turn, turn_step, start, grid, road_dir, junction, visited):
    """Next cell of a route; ``turn`` is 0 straight, 1 right, 2 left."""
    h, w = grid.shape
    turn_completed = False
    turn_prog = False
    neigh = []
    for m in _MOVES:
        n = (curr[0] + m[0], curr[1] + m[1])
        if not (0 <= n[0] <= h - 1 and 0 <= n[1] <= w - 1 and grid[n]):
            continue
        if n in visited:
            continue
        if junction[n] == junction[curr] == 1:
            if (turn == 0 or turn == 2) and (n[0] == start[0] or n[1] == start[1]):
                neigh.append(n)
                if turn == 2:
                    turn_prog = True
            elif turn == 2 and turn_step == 1:
                neigh.append(n)
                turn_prog = True
        elif (
            junction[curr]
            and not junction[n]
            and turn == 2
            and turn_step == 2
            and (abs(start[0] - n[0]) == 2 or abs(start[1] - n[1]) == 2)
        ):
            neigh.append(n)
            turn_completed = True
        elif junction[n] and not junction[curr]:
            neigh.append(n)
        elif turn == 1 and not junction[n] and junction[curr]:
            neigh.append(n)
            turn_completed = True
        elif turn == 0 and junction[curr] and road_dir[n] == road_dir[start]:
            neigh.append(n)
            turn_completed = True
        elif road_dir[n] == road_dir[curr] and not junction[curr]:
            neigh.append(n)
    if not neigh:
        raise RuntimeError(f"No route continues from {curr}")
    return neigh[0], turn_prog, turn_completed


def _routes(dims, route_grid, difficulty: str) -> list:
    """Routes grouped by arrival point, each an array of (row, col) cells."""
    h, w = dims
    if difficulty == "easy":
        return [
            [np.array([(i, w // 2) for i in range(h)])],
            [np.array([(h // 2, i) for i in range(w)])],
        ]
    arrival, finish, road_dir, junction = _junction_layout(dims, route_grid, difficulty)
    n_turn1 = 3
    n_turn2 = 1 if difficulty == "medium" else 3
    routes = []
    for i, start_point in enumerate(arrival):
        exits = finish[:i] + finish[i + 1 :]
        paths = []
        for turn_1 in range(n_turn1):
            for turn_2 in range(n_turn2):
                total_turns = 0
                curr_turn = turn_1
                current = start = start_point
                path = [current]
                visited = set()
                turn_step = 0
                while current not in exits:
                    visited.add(current)
                    current, turn_prog, turn_completed = _next_move(
                        current,
                        curr_turn,
                        turn_step,
                        start,
                        route_grid,
                        road_dir,
                        junction,
                        visited,
                    )
                    if curr_turn == 2 and turn_prog:
                        turn_step += 1
                    if turn_completed:
                        total_turns += 1
                        curr_turn = turn_2
                        turn_step = 0
                        start = current
                    if total_turns == 2:
                        curr_turn = 0
                    path.append(current)
                paths.append(np.array(path))
                if total_turns == 1:
                    break
        routes.append(paths)
    return routes


@struct.dataclass
class State(BaseState):
    """State of the Traffic Junction environment.

    Attributes:
        car_loc: (row, col) of each car; ``(0, 0)`` while a car is outside.
        alive: Whether each car is currently on the road.
        route_pos: Index of each car's cell along its route.
        route: Index of each car's route in ``TrafficJunctionGrid.routes``.
        last_action: Each car's last action while on the road.
        wait: Steps each car has spent on the road.
        has_failed: Whether any collision happened this episode.
    """

    car_loc: Int[Array, "num_agents 2"]
    alive: Bool[Array, " num_agents"]
    route_pos: Int[Array, " num_agents"]
    route: Int[Array, " num_agents"]
    last_action: Int[Array, " num_agents"]
    wait: Int[Array, " num_agents"]
    has_failed: Bool[Array, ""]


class TrafficJunctionGrid(MultiAgentEnv):
    """Grid Traffic Junction with easy, medium and hard layouts.

    Each car on the road receives ``-0.01 * wait`` per step, where ``wait`` is
    the number of steps it has spent on the road, and ``-10`` for every step in
    which it shares a cell with another car. Cars off the road receive 0.
    Episodes end after ``max_steps``; ``info["success"]`` is true when no
    collision has happened in the episode.

    ``difficulty`` selects the paper's layout and its defaults for the other
    arguments, which can be overridden individually: easy is one junction of
    two one-way roads on a 7x7 grid, medium one junction of two-way roads on
    14x14, and hard four junctions of two-way roads on 18x18. The paper raises
    ``add_rate`` with a curriculum during training; the defaults are its final
    values.

    Args:
        difficulty: ``"easy"``, ``"medium"`` or ``"hard"``.
        num_agents: Maximum number of cars on the road at once.
        dim: Road length. Easy uses a ``(dim + 1)``-sized grid, as in the
            reference.
        vision: Radius of each car's square observation window.
        max_steps: Episode length limit.
        add_rate: Probability of a car entering at each arrival point per step.
    """

    def __init__(
        self,
        difficulty: str = "easy",
        num_agents: Optional[int] = None,
        dim: Optional[int] = None,
        vision: int = 0,
        max_steps: Optional[int] = None,
        add_rate: Optional[float] = None,
    ) -> None:
        if difficulty not in DIFFICULTIES:
            raise ValueError(
                f"difficulty must be one of {tuple(DIFFICULTIES)}, got {difficulty!r}"
            )
        defaults = DIFFICULTIES[difficulty]
        num_agents = defaults["num_agents"] if num_agents is None else num_agents
        dim = defaults["dim"] if dim is None else dim
        max_steps = defaults["max_steps"] if max_steps is None else max_steps
        add_rate = defaults["add_rate"] if add_rate is None else add_rate
        if difficulty in ("easy", "medium"):
            if dim % 2 != 0 or dim < 4 + vision:
                raise ValueError(f"{difficulty} needs an even dim >= 4 + vision")
        elif dim < 9 or dim % 3 != 0:
            raise ValueError("hard needs dim >= 9 and a multiple of 3")
        super().__init__(num_agents)

        self.difficulty = difficulty
        self.dim = dim
        self.vision = vision
        self.max_steps = max_steps
        self.add_rate = add_rate
        self.num_actions = 2

        self.TIMESTEP_PENALTY = -0.01
        self.CRASH_PENALTY = -10.0

        self.dims = (dim + 1, dim + 1) if difficulty == "easy" else (dim, dim)
        h, w = self.dims
        # One vocabulary id per road cell, then outside, (unused) road and car
        # classes, as in the reference's "bool" vocabulary. The reference sizes
        # the road ids from dim rather than the enlarged easy grid, which gives
        # two easy road cells the outside and road ids; this counts them all.
        nroad = {"easy": 1, "medium": 2, "hard": 4}[difficulty]
        self.base = nroad * (h + w)
        self.outside_class = self.base
        self.car_class = self.base + 2
        self.vocab_size = self.base + 3
        self.vis_size = 2 * vision + 1
        self.obs_dim = 2 + self.vis_size * self.vis_size * self.vocab_size

        grid = np.full(self.dims, self.outside_class, dtype=np.int32)
        blocks = _road_blocks(w, h, difficulty)
        for block in blocks:
            grid[block] = 1
        route_grid = grid.copy()
        start = 0
        for block in blocks:
            size = grid[block].size
            grid[block] = np.arange(start, start + size).reshape(grid[block].shape)
            start += size
        padded = np.pad(grid, vision, constant_values=self.outside_class)
        self.empty_grid = jax.nn.one_hot(jnp.asarray(padded), self.vocab_size)

        groups = _routes(self.dims, route_grid, difficulty)
        routes = [r for group in groups for r in group]
        self.num_routes = len(routes)
        self.num_arrivals = len(groups)
        self.routes_per_arrival = len(groups[0])
        assert all(len(g) == self.routes_per_arrival for g in groups)
        route_len = max(len(r) for r in routes)
        padded_routes = np.zeros((self.num_routes, route_len, 2), dtype=np.int32)
        for i, r in enumerate(routes):
            padded_routes[i, : len(r)] = r
        self.routes = jnp.asarray(padded_routes)
        self.route_lengths = jnp.asarray([len(r) for r in routes], dtype=jnp.int32)

        # Values are 0/1 apart from car counts, which can reach num_agents.
        self.observation_spaces = {
            a: Box(0.0, float(num_agents), (self.obs_dim,)) for a in self.agents
        }
        self.action_spaces = {a: Discrete(self.num_actions) for a in self.agents}

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: PRNGKeyArray) -> Tuple[Observations, State]:
        """Start with an empty road; cars enter from the first step."""
        n = self.num_agents
        state = State(
            car_loc=jnp.zeros((n, 2), dtype=jnp.int32),
            alive=jnp.zeros(n, dtype=jnp.bool_),
            route_pos=jnp.zeros(n, dtype=jnp.int32),
            route=jnp.zeros(n, dtype=jnp.int32),
            last_action=jnp.zeros(n, dtype=jnp.int32),
            wait=jnp.zeros(n, dtype=jnp.int32),
            has_failed=jnp.array(False),
            done=jnp.array(False),
            step=jnp.array(0),
        )
        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self, key: PRNGKeyArray, state: State, actions: Actions
    ) -> Tuple[Observations, State, Rewards, Dones, Infos]:
        """Move the cars, let new cars enter, then score collisions."""
        # The baselines pass each action as a scalar or a (1,) array.
        acts = jnp.array([actions[a] for a in self.agents]).reshape(self.num_agents)
        state = self._move_cars(state, acts)
        state = self._add_cars(key, state)

        same_cell = jnp.all(state.car_loc[:, None] == state.car_loc[None, :], axis=-1)
        same_cell = same_cell & ~jnp.eye(self.num_agents, dtype=jnp.bool_)
        # Cars off the road wait at (0, 0), which is never road.
        crashed = jnp.any(same_cell, axis=1) & jnp.any(state.car_loc != 0, axis=-1)
        reward = self.TIMESTEP_PENALTY * state.wait + self.CRASH_PENALTY * crashed
        reward = jnp.where(state.alive, reward, 0.0)

        step = state.step + 1
        done = step >= self.max_steps
        state = state.replace(
            has_failed=state.has_failed | jnp.any(crashed), done=done, step=step
        )
        rewards = {a: reward[i] for i, a in enumerate(self.agents)}
        dones = {a: done for a in self.agents}
        dones["__all__"] = done
        infos = {"success": jnp.full((self.num_agents,), ~state.has_failed)}
        return self.get_obs(state), state, rewards, dones, infos

    def _move_cars(self, state: State, acts: Int[Array, " num_agents"]) -> State:
        """GAS advances a car along its route; a car past its end leaves."""
        gas = state.alive & (acts == GAS)
        route_pos = jnp.where(gas, state.route_pos + 1, state.route_pos)
        finished = gas & (route_pos >= self.route_lengths[state.route])
        next_loc = self.routes[
            state.route, jnp.minimum(route_pos, self.routes.shape[1] - 1)
        ]
        car_loc = jnp.where((gas & ~finished)[:, None], next_loc, state.car_loc)
        car_loc = jnp.where(finished[:, None], 0, car_loc)
        wait = jnp.where(state.alive, state.wait + 1, state.wait)
        return state.replace(
            car_loc=car_loc,
            alive=state.alive & ~finished,
            route_pos=route_pos,
            # As in the reference, a car keeps its previous last action when it
            # leaves, and shows it again if it re-enters.
            last_action=jnp.where(state.alive & ~finished, acts, state.last_action),
            wait=jnp.where(finished, 0, wait),
        )

    def _add_cars(self, key: PRNGKeyArray, state: State) -> State:
        """At each arrival point, a car enters with probability ``add_rate``."""

        def _arrival(state, xs):
            arrival, key = xs
            key_add, key_car, key_route = jax.random.split(key, 3)
            add = (jax.random.uniform(key_add) <= self.add_rate) & ~jnp.all(state.alive)
            # A random car that is off the road, as in the reference.
            car = jax.random.categorical(key_car, jnp.where(state.alive, -jnp.inf, 0.0))
            route = arrival * self.routes_per_arrival + jax.random.randint(
                key_route, (), 0, self.routes_per_arrival
            )

            def _set(x, value):
                return x.at[car].set(jnp.where(add, value, x[car]))

            state = state.replace(
                car_loc=_set(state.car_loc, self.routes[route, 0]),
                alive=_set(state.alive, True),
                route_pos=_set(state.route_pos, 0),
                route=_set(state.route, route),
            )
            return state, None

        keys = jax.random.split(key, self.num_arrivals)
        state, _ = jax.lax.scan(_arrival, state, (jnp.arange(self.num_arrivals), keys))
        return state

    def get_obs(self, state: State) -> Observations:
        """Per car: last action, route id and the flattened vision window.

        The two scalars are normalised to ``[0, 1]``. Cars off the road observe
        zeros.
        """
        grid = self.empty_grid.at[
            state.car_loc[:, 0] + self.vision,
            state.car_loc[:, 1] + self.vision,
            self.car_class,
        ].add(state.alive.astype(jnp.float32))

        def _car_obs(loc, last_action, route, alive):
            window = jax.lax.dynamic_slice(
                grid,
                (loc[0], loc[1], 0),
                (self.vis_size, self.vis_size, self.vocab_size),
            )
            scalars = jnp.array(
                [
                    last_action / (self.num_actions - 1),
                    route / max(self.num_routes - 1, 1),
                ],
                dtype=jnp.float32,
            )
            obs = jnp.concatenate([scalars, window.reshape(-1)])
            return jnp.where(alive, obs, 0.0)

        obs = jax.vmap(_car_obs)(
            state.car_loc, state.last_action, state.route, state.alive
        )
        return {a: obs[i] for i, a in enumerate(self.agents)}

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: State) -> dict:
        """Both actions are always available."""
        return {a: jnp.ones(self.num_actions, dtype=jnp.bool_) for a in self.agents}

    @property
    def agent_classes(self) -> dict:
        return {"car": self.agents}

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
        """Draw the roads and cars; returns the artists to update."""
        import numpy as np
        from matplotlib.patches import Circle, Rectangle

        h, w = self.dims
        ax.clear()
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(-0.5, h - 0.5)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_facecolor("#f5f5f5")
        ax.set_title(f"Traffic Junction ({self.difficulty})  step {int(state.step)}")

        road = np.array(self.empty_grid[:, :, self.outside_class] == 0)
        road = road[self.vision : self.vision + h, self.vision : self.vision + w]
        for r, c in zip(*np.nonzero(road)):
            ax.add_patch(
                Rectangle(
                    (c - 0.5, r - 0.5),
                    1,
                    1,
                    facecolor="#d0d0d0",
                    edgecolor="#bbbbbb",
                    lw=0.3,
                )
            )

        locs = np.array(state.car_loc)
        alive = np.array(state.alive)
        patches, labels = [], []
        for i in range(self.num_agents):
            r, c = int(locs[i, 0]), int(locs[i, 1])
            circle = Circle(
                (c, r),
                0.3,
                color=self._COLORS[i % len(self._COLORS)],
                ec="black",
                lw=1.0,
                zorder=10,
                visible=bool(alive[i]),
            )
            ax.add_patch(circle)
            label = ax.text(
                c,
                r,
                str(i),
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
                color="white",
                zorder=11,
                visible=bool(alive[i]),
            )
            patches.append(circle)
            labels.append(label)
        return {"ax": ax, "patches": patches, "labels": labels}

    def update_render(self, artists, state: State):
        """Move the artists returned by ``init_render`` to ``state``."""
        import numpy as np

        artists["ax"].set_title(
            f"Traffic Junction ({self.difficulty})  step {int(state.step)}"
        )
        locs = np.array(state.car_loc)
        alive = np.array(state.alive)
        for i in range(self.num_agents):
            r, c = int(locs[i, 0]), int(locs[i, 1])
            artists["patches"][i].center = (c, r)
            artists["patches"][i].set_visible(bool(alive[i]))
            artists["labels"][i].set_position((c, r))
            artists["labels"][i].set_visible(bool(alive[i]))
        return artists

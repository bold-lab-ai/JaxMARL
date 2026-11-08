# Reference implementation from IC3Net, used only to test the JAX port against it.
# Source: https://github.com/IC3Net/IC3Net/tree/master/ic3net-envs (MIT License,
# Copyright (c) 2019 Amanpreet Singh, Tushar Jain and Sainbayar Sukhbaatar).
# Changes: removed the gym, gym.spaces and curses dependencies and rendering,
# made np.ogrid output a list and np.row_stack np.vstack for NumPy 2, and import
# traffic_helper locally.
# ruff: noqa
# fmt: off
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simulate a traffic junction environment.
Each agent can observe itself (it's own identity) i.e. s_j = j and vision, path ahead of it.

Design Decisions:
    - Memory cheaper than time (compute)
    - Using Vocab for class of box:
    - Action Space & Observation Space are according to an agent
    - Rewards
         -0.05 at each time step till the time
         -10 for each crash
    - Episode ends when all cars reach destination / max steps
    - Obs. State:
"""

# core modules
import random
import math

# 3rd party modules
import numpy as np
from ic3net_traffic_helper import *


def nPr(n,r):
    f = math.factorial
    return f(n)//f(n-r)

class TrafficJunctionEnv(object):
    # metadata = {'render.modes': ['human']}

    def __init__(self,):
        self.__version__ = "0.0.1"

        # TODO: better config handling
        self.OUTSIDE_CLASS = 0
        self.ROAD_CLASS = 1
        self.CAR_CLASS = 2
        self.TIMESTEP_PENALTY = -0.01
        self.CRASH_PENALTY = -10

        self.episode_over = False
        self.has_failed = 0


    def init_args(self, parser):
        env = parser.add_argument_group('Traffic Junction task')
        env.add_argument('--dim', type=int, default=5,
                         help="Dimension of box (i.e length of road) ")
        env.add_argument('--vision', type=int, default=1,
                         help="Vision of car")
        env.add_argument('--add_rate_min', type=float, default=0.05,
                         help="rate at which to add car (till curr. start)")
        env.add_argument('--add_rate_max', type=float, default=0.2,
                         help=" max rate at which to add car")
        env.add_argument('--curr_start', type=float, default=0,
                         help="start making harder after this many epochs [0]")
        env.add_argument('--curr_end', type=float, default=0,
                         help="when to make the game hardest [0]")
        env.add_argument('--difficulty', type=str, default='easy',
                         help="Difficulty level, easy|medium|hard")
        env.add_argument('--vocab_type', type=str, default='bool',
                         help="Type of location vector to use, bool|scalar")


    def multi_agent_init(self, args):
        # General variables defining the environment : CONFIG
        params = ['dim', 'vision', 'add_rate_min', 'add_rate_max', 'curr_start', 'curr_end',
                  'difficulty', 'vocab_type']

        for key in params:
            setattr(self, key, getattr(args, key))

        self.ncar = args.nagents
        self.dims = dims = (self.dim, self.dim)
        difficulty = args.difficulty
        vision = args.vision

        if difficulty in ['medium','easy']:
            assert dims[0]%2 == 0, 'Only even dimension supported for now.'

            assert dims[0] >= 4 + vision, 'Min dim: 4 + vision'

        if difficulty == 'hard':
            assert dims[0] >= 9, 'Min dim: 9'
            assert dims[0]%3 ==0, 'Hard version works for multiple of 3. dim. only.'

        # Add rate
        self.exact_rate = self.add_rate = self.add_rate_min
        self.epoch_last_update = 0

        # Define what an agent can do -
        # (0: GAS, 1: BRAKE) i.e. (0: Move 1-step, 1: STAY)
        self.naction = 2
        self.action_space = None

        # make no. of dims odd for easy case.
        if difficulty == 'easy':
            self.dims = list(dims)
            for i in range(len(self.dims)):
                self.dims[i] += 1

        nroad = {'easy':2,
                'medium':4,
                'hard':8}

        dim_sum = dims[0] + dims[1]
        base = {'easy':   dim_sum,
                'medium': 2 * dim_sum,
                'hard':   4 * dim_sum}

        self.npath = nPr(nroad[difficulty],2)

        # Setting max vocab size for 1-hot encoding
        if self.vocab_type == 'bool':
            self.BASE = base[difficulty]
            self.OUTSIDE_CLASS += self.BASE
            self.CAR_CLASS += self.BASE
            # car_type + base + outside + 0-index
            self.vocab_size = 1 + self.BASE + 1 + 1
            self.observation_space = None
        else:
            # r_i, (x,y), vocab = [road class + car]
            self.vocab_size = 1 + 1

            # Observation for each agent will be 4-tuple of (r_i, last_act, len(dims), vision * vision * vocab)
            self.observation_space = None
            # Actual observation will be of the shape 1 * ncar * ((x,y) , (2v+1) * (2v+1) * vocab_size)

        self._set_grid()

        if difficulty == 'easy':
            self._set_paths_easy()
        else:
            self._set_paths(difficulty)

        return

    def reset(self, epoch=None):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.episode_over = False
        self.has_failed = 0

        self.alive_mask = np.zeros(self.ncar)
        self.wait = np.zeros(self.ncar)
        self.cars_in_sys = 0

        # Chosen path for each car:
        self.chosen_path = [0] * self.ncar
        # when dead => no route, must be masked by trainer.
        self.route_id = [-1] * self.ncar

        # self.cars = np.zeros(self.ncar)
        # Current car to enter system
        # self.car_i = 0
        # Ids i.e. indexes
        self.car_ids = np.arange(self.CAR_CLASS,self.CAR_CLASS + self.ncar)

        # Starting loc of car: a place where everything is outside class
        self.car_loc = np.zeros((self.ncar, len(self.dims)),dtype=int)
        self.car_last_act = np.zeros(self.ncar, dtype=int) # last act GAS when awake

        self.car_route_loc = np.full(self.ncar, - 1)

        # stat - like success ratio
        self.stat = dict()

        # set add rate according to the curriculum
        epoch_range = (self.curr_end - self.curr_start)
        add_rate_range = (self.add_rate_max - self.add_rate_min)
        if epoch is not None and epoch_range > 0 and add_rate_range > 0 and epoch > self.epoch_last_update:
            self.curriculum(epoch)
            self.epoch_last_update = epoch

        # Observation will be ncar * vision * vision ndarray
        obs = self._get_obs()
        return obs

    def step(self, action):
        """
        The agents(car) take a step in the environment.

        Parameters
        ----------
        action : shape - either ncar or ncar x 1

        Returns
        -------
        obs, reward, episode_over, info : tuple
            obs (object) :
            reward (ncar x 1) : PENALTY for each timestep when in sys & CRASH PENALTY on crashes.
            episode_over (bool) : Will be true when episode gets over.
            info (dict) : diagnostic information useful for debugging.
        """
        if self.episode_over:
            raise RuntimeError("Episode is done")

        # Expected shape: either ncar or ncar x 1
        action = np.array(action).squeeze()

        assert np.all(action <= self.naction), "Actions should be in the range [0,naction)."

        assert len(action) == self.ncar, "Action for each agent should be provided."

        # No one is completed before taking action
        self.is_completed = np.zeros(self.ncar)

        for i, a in enumerate(action):
            self._take_action(i, a)

        self._add_cars()

        obs = self._get_obs()
        reward = self._get_reward()

        debug = {'car_loc':self.car_loc,
                'alive_mask': np.copy(self.alive_mask),
                'wait': self.wait,
                'cars_in_sys': self.cars_in_sys,
                'is_completed': np.copy(self.is_completed)}

        self.stat['success'] = 1 - self.has_failed
        self.stat['add_rate'] = self.add_rate

        return obs, reward, self.episode_over, debug



    def seed(self):
        return

    def _set_grid(self):
        self.grid = np.full(self.dims[0] * self.dims[1], self.OUTSIDE_CLASS, dtype=int).reshape(self.dims)
        w, h = self.dims

        # Mark the roads
        roads = get_road_blocks(w,h, self.difficulty)
        for road in roads:
            self.grid[road] = self.ROAD_CLASS
        if self.vocab_type == 'bool':
            self.route_grid = self.grid.copy()
            start = 0
            for road in roads:
                sz = int(np.prod(self.grid[road].shape))
                self.grid[road] = np.arange(start, start + sz).reshape(self.grid[road].shape)
                start += sz

        # Padding for vision
        self.pad_grid = np.pad(self.grid, self.vision, 'constant', constant_values = self.OUTSIDE_CLASS)

        self.empty_bool_base_grid = self._onehot_initialization(self.pad_grid)

    def _get_obs(self):
        h, w = self.dims
        self.bool_base_grid = self.empty_bool_base_grid.copy()

        # Mark cars' location in Bool grid
        for i, p in enumerate(self.car_loc):
            self.bool_base_grid[p[0] + self.vision, p[1] + self.vision, self.CAR_CLASS] += 1


        # remove the outside class.
        if self.vocab_type == 'scalar':
            self.bool_base_grid = self.bool_base_grid[:,:,1:]


        obs = []
        for i, p in enumerate(self.car_loc):
            # most recent action
            act = self.car_last_act[i] / (self.naction - 1)

            # route id
            r_i = self.route_id[i] / (self.npath - 1)

            # loc
            p_norm = p / (h-1, w-1)

            # vision square
            slice_y = slice(p[0], p[0] + (2 * self.vision) + 1)
            slice_x = slice(p[1], p[1] + (2 * self.vision) + 1)
            v_sq = self.bool_base_grid[slice_y, slice_x]

            # when dead, all obs are 0. But should be masked by trainer.
            if self.alive_mask[i] == 0:
                act = np.zeros_like(act)
                r_i = np.zeros_like(r_i)
                p_norm = np.zeros_like(p_norm)
                v_sq = np.zeros_like(v_sq)

            if self.vocab_type == 'bool':
                o = tuple((act, r_i, v_sq))
            else:
                o = tuple((act, r_i, p_norm, v_sq))
            obs.append(o)

        obs = tuple(obs)

        return obs


    def _add_cars(self):
        for r_i, routes in enumerate(self.routes):
            if self.cars_in_sys >= self.ncar:
                return

            # Add car to system and set on path
            if np.random.uniform() <= self.add_rate:

                # chose dead car on random
                idx = self._choose_dead()
                # make it alive
                self.alive_mask[idx] = 1

                # choose path randomly & set it
                p_i = np.random.choice(len(routes))
                # make sure all self.routes have equal len/ same no. of routes
                self.route_id[idx] = p_i + r_i * len(routes)
                self.chosen_path[idx] = routes[p_i]

                # set its start loc
                self.car_route_loc[idx] = 0
                self.car_loc[idx] = routes[p_i][0]

                # increase count
                self.cars_in_sys += 1

    def _set_paths_easy(self):
        h, w = self.dims
        self.routes = {
            'TOP': [],
            'LEFT': []
        }

        # 0 refers to UP to DOWN, type 0
        full = [(i, w//2) for i in range(h)]
        self.routes['TOP'].append(np.array([*full]))

        # 1 refers to LEFT to RIGHT, type 0
        full = [(h//2, i) for i in range(w)]
        self.routes['LEFT'].append(np.array([*full]))

        self.routes = list(self.routes.values())


    def _set_paths_medium_old(self):
        h,w = self.dims
        self.routes = {
            'TOP': [],
            'LEFT': [],
            'RIGHT': [],
            'DOWN': []
        }

        # type 0 paths: go straight on junction
        # type 1 paths: take right on junction
        # type 2 paths: take left on junction


        # 0 refers to UP to DOWN, type 0
        full = [(i, w//2-1) for i in range(h)]
        self.routes['TOP'].append(np.array([*full]))

        # 1 refers to UP to LEFT, type 1
        first_half = full[:h//2]
        second_half = [(h//2 - 1, i) for i in range(w//2 - 2,-1,-1) ]
        self.routes['TOP'].append(np.array([*first_half, *second_half]))

        # 2 refers to UP to RIGHT, type 2
        second_half = [(h//2, i) for i in range(w//2-1, w) ]
        self.routes['TOP'].append(np.array([*first_half, *second_half]))


        # 3 refers to LEFT to RIGHT, type 0
        full = [(h//2, i) for i in range(w)]
        self.routes['LEFT'].append(np.array([*full]))

        # 4 refers to LEFT to DOWN, type 1
        first_half = full[:w//2]
        second_half = [(i, w//2 - 1) for i in range(h//2+1, h)]
        self.routes['LEFT'].append(np.array([*first_half, *second_half]))

        # 5 refers to LEFT to UP, type 2
        second_half = [(i, w//2) for i in range(h//2, -1,-1) ]
        self.routes['LEFT'].append(np.array([*first_half, *second_half]))


        # 6 refers to DOWN to UP, type 0
        full = [(i, w//2) for i in range(h-1,-1,-1)]
        self.routes['DOWN'].append(np.array([*full]))

        # 7 refers to DOWN to RIGHT, type 1
        first_half = full[:h//2]
        second_half = [(h//2, i) for i in range(w//2+1,w)]
        self.routes['DOWN'].append(np.array([*first_half, *second_half]))

        # 8 refers to DOWN to LEFT, type 2
        second_half = [(h//2-1, i) for i in range(w//2,-1,-1)]
        self.routes['DOWN'].append(np.array([*first_half, *second_half]))


        # 9 refers to RIGHT to LEFT, type 0
        full = [(h//2-1, i) for i in range(w-1,-1,-1)]
        self.routes['RIGHT'].append(np.array([*full]))

        # 10 refers to RIGHT to UP, type 1
        first_half = full[:w//2]
        second_half = [(i, w//2) for i in range(h//2 -2, -1,-1)]
        self.routes['RIGHT'].append(np.array([*first_half, *second_half]))

        # 11 refers to RIGHT to DOWN, type 2
        second_half = [(i, w//2-1) for i in range(h//2-1, h)]
        self.routes['RIGHT'].append(np.array([*first_half, *second_half]))


        # PATHS_i: 0 to 11
        # 0 refers to UP to down,
        # 1 refers to UP to left,
        # 2 refers to UP to right,
        # 3 refers to LEFT to right,
        # 4 refers to LEFT to down,
        # 5 refers to LEFT to up,
        # 6 refers to DOWN to up,
        # 7 refers to DOWN to right,
        # 8 refers to DOWN to left,
        # 9 refers to RIGHT to left,
        # 10 refers to RIGHT to up,
        # 11 refers to RIGHT to down,

        # Convert to routes dict to list of paths
        paths = []
        for r in self.routes.values():
            for p in r:
                paths.append(p)

        # Check number of paths
        # assert len(paths) == self.npath

        # Test all paths
        assert self._unittest_path(paths)

    def _set_paths(self, difficulty):
        route_grid = self.route_grid if self.vocab_type == 'bool' else self.grid
        self.routes = get_routes(self.dims, route_grid, difficulty)

        # Convert/unroll routes which is a list of list of paths
        paths = []
        for r in self.routes:
            for p in r:
                paths.append(p)

        # Check number of paths
        assert len(paths) == self.npath

        # Test all paths
        assert self._unittest_path(paths)


    def _unittest_path(self,paths):
        for i, p in enumerate(paths[:-1]):
            next_dif = p - np.vstack([p[1:], p[-1]])
            next_dif = np.abs(next_dif[:-1])
            step_jump = np.sum(next_dif, axis =1)
            if np.any(step_jump != 1):
                print("Any", p, i)
                return False
            if not np.all(step_jump == 1):
                print("All", p, i)
                return False
        return True


    def _take_action(self, idx, act):
        # non-active car
        if self.alive_mask[idx] == 0:
            return

        # add wait time for active cars
        self.wait[idx] += 1

        # action BRAKE i.e STAY
        if act == 1:
            self.car_last_act[idx] = 1
            return

        # GAS or move
        if act==0:
            prev = self.car_route_loc[idx]
            self.car_route_loc[idx] += 1
            curr = self.car_route_loc[idx]

            # car/agent has reached end of its path
            if curr == len(self.chosen_path[idx]):
                self.cars_in_sys -= 1
                self.alive_mask[idx] = 0
                self.wait[idx] = 0

                # put it at dead loc
                self.car_loc[idx] = np.zeros(len(self.dims),dtype=int)
                self.is_completed[idx] = 1
                return

            elif curr > len(self.chosen_path[idx]):
                print(curr)
                raise RuntimeError("Out of boud car path")

            prev = self.chosen_path[idx][prev]
            curr = self.chosen_path[idx][curr]

            # assert abs(curr[0] - prev[0]) + abs(curr[1] - prev[1]) == 1 or curr_path = 0
            self.car_loc[idx] = curr

            # Change last act for color:
            self.car_last_act[idx] = 0



    def _get_reward(self):
        reward = np.full(self.ncar, self.TIMESTEP_PENALTY) * self.wait

        for i, l in enumerate(self.car_loc):
            if (len(np.where(np.all(self.car_loc[:i] == l,axis=1))[0]) or \
               len(np.where(np.all(self.car_loc[i+1:] == l,axis=1))[0])) and l.any():
               reward[i] += self.CRASH_PENALTY
               self.has_failed = 1

        reward = self.alive_mask * reward
        return reward

    def _onehot_initialization(self, a):
        if self.vocab_type == 'bool':
            ncols = self.vocab_size
        else:
            ncols = self.vocab_size + 1 # 1 is for outside class which will be removed later.
        out = np.zeros(a.shape + (ncols,), dtype=int)
        out[self._all_idx(a, axis=2)] = 1
        return out

    def _all_idx(self, idx, axis):
        grid = list(np.ogrid[tuple(map(slice, idx.shape))])
        grid.insert(axis, idx)
        return tuple(grid)

    def reward_terminal(self):
        return np.zeros_like(self._get_reward())

    def _choose_dead(self):
        # all idx
        car_idx = np.arange(len(self.alive_mask))
        # random choice of idx from dead ones.
        return np.random.choice(car_idx[self.alive_mask == 0])

    def curriculum(self, epoch):
        step_size = 0.01
        step = (self.add_rate_max - self.add_rate_min) / (self.curr_end - self.curr_start)

        if self.curr_start <= epoch < self.curr_end:
            self.exact_rate = self.exact_rate + step
            self.add_rate = step_size * (self.exact_rate // step_size)

# Reference implementation from IC3Net, used only to test the JAX port against it.
# Source: https://github.com/IC3Net/IC3Net/tree/master/ic3net-envs (MIT License,
# Copyright (c) 2019 Amanpreet Singh, Tushar Jain and Sainbayar Sukhbaatar).
# Changes: replaced the deprecated `grid.dtype = int` with `grid = grid.view(int)`.
# ruff: noqa
# fmt: off
import numpy as np

move = [(-1,0),(1,0),(0,-1),(0,1)]

def get_road_blocks(w, h, difficulty):

    # assuming 1 is the lane width for each direction.
    road_blocks = {
        'easy':   [ np.s_[h//2, :],
                    np.s_[:, w//2]],

        'medium': [ np.s_[h//2 - 1 : h//2 + 1, :],
                    np.s_[:, w//2 - 1 : w//2 + 1]],

        'hard':   [ np.s_[h//3-2: h//3, :],
                    np.s_[2* h//3: 2* h//3 + 2, :],

                    np.s_[:, w//3-2: w//3],
                    np.s_[:, 2* h//3: 2* h//3 + 2]],
    }

    return road_blocks[difficulty]

def goal_reached(place_i, curr, finish_points):
    return curr in finish_points[:place_i] + finish_points[place_i+1:]


def get_add_mat(dims, grid, difficulty):
    h,w = dims

    road_dir = grid.copy()
    junction = np.zeros_like(grid)

    if difficulty == 'medium':
        arrival_points = [  (0, w//2-1),    # TOP
                            (h-1,w//2),     # BOTTOM
                            (h//2, 0),      # LEFT
                            (h//2-1,w-1)]   # RIGHT

        finish_points =  [  (0, w//2),      # TOP
                            (h-1,w//2-1),   # BOTTOM
                            (h//2-1, 0),    # LEFT
                            (h//2,w-1)]     # RIGHT

        # mark road direction
        road_dir[h//2, :] = 2
        road_dir[h//2 - 1, :] = 3
        road_dir[:, w//2 ] = 4

        # mark the Junction
        junction[h//2-1:h//2+1,w//2-1:w//2+1 ] =1

    elif difficulty =='hard':
        arrival_points = [  (0, w//3-2),    # TOP-left
                            (0,2*w//3),     # TOP-right

                            (h//3-1, 0),    # LEFT-top
                            (2*h//3+1,0),   # LEFT-bottom

                            (h-1,w//3-1),   # BOTTOM-left
                            (h-1,2*w//3+1), # BOTTOM-right

                            (h//3-2, w-1),  # RIGHT-top
                            (2*h//3,w-1)]   # RIGHT-bottom


        finish_points = [  (0, w//3-1),     # TOP-left
                            (0,2*w//3+1),   # TOP-right

                            (h//3-2, 0),    # LEFT-top
                            (2*h//3,0),     # LEFT-bottom

                            (h-1,w//3-2),   # BOTTOM-left
                            (h-1,2*w//3),   # BOTTOM-right

                            (h//3-1, w-1),  # RIGHT-top
                            (2*h//3+1,w-1)] # RIGHT-bottom

        # mark road direction
        road_dir[h//3-1, :] = 2
        road_dir[2*h//3, :] = 3
        road_dir[2*h//3 + 1, :] = 4

        road_dir[:, w//3-2 ] = 5
        road_dir[:, w//3-1 ] = 6
        road_dir[:, 2*w//3 ] = 7
        road_dir[:, 2*w//3 +1] = 8

        # mark the Junctions
        junction[h//3-2:h//3, w//3-2:w//3 ] = 1
        junction[2*h//3:2*h//3+2, w//3-2:w//3 ] = 1

        junction[h//3-2:h//3, 2*w//3:2*w//3+2 ] = 1
        junction[2*h//3:2*h//3+2, 2*w//3:2*w//3+2 ] = 1

    return arrival_points, finish_points, road_dir, junction


def next_move(curr, turn, turn_step, start, grid, road_dir, junction, visited):
    h,w = grid.shape
    turn_completed = False
    turn_prog = False
    neigh =[]
    for m in move:
        # check lane while taking left turn
        n = (curr[0] + m[0], curr[1] + m[1])
        if 0 <= n[0] <= h-1 and 0 <= n[1] <= w-1 and grid[n] and n not in visited:
            # On Junction, use turns
            if junction[n] == junction[curr] == 1:
                if (turn == 0 or turn == 2) and ((n[0] == start[0]) or (n[1] == start[1])):
                    # Straight on junction for either left or straight
                    neigh.append(n)
                    if turn == 2:
                        turn_prog = True

                # left from junction
                elif turn == 2 and turn_step ==1:
                    neigh.append(n)
                    turn_prog = True

                else:
                    # End of path
                    pass

            # Completing left turn on junction
            elif junction[curr] and not junction[n] and turn ==2 and turn_step==2 \
                and (abs(start[0] - n[0]) ==2 or abs(start[1] - n[1]) ==2):
                neigh.append(n)
                turn_completed =True

            # junction seen, get onto it;
            elif (junction[n] and not junction[curr]):
                neigh.append(n)

            # right from junction
            elif turn == 1 and not junction[n] and junction[curr]:
                neigh.append(n)
                turn_completed =True

            # Straight from jucntion
            elif turn == 0 and junction[curr] and road_dir[n] == road_dir[start]:
                neigh.append(n)
                turn_completed = True

            # keep going no decision to make;
            elif road_dir[n] == road_dir[curr] and not junction[curr]:
                neigh.append(n)

    if neigh:
        return neigh[0], turn_prog, turn_completed
    if len(neigh) != 1:
        raise RuntimeError("next move should be of len 1. Reached ambiguous situation.")



def get_routes(dims, grid, difficulty):
    '''
    returns
        - routes: type list of list
        list for each arrival point of list of routes from that arrival point.
    '''
    grid = grid.view(int)
    h,w = dims

    assert difficulty == 'medium' or difficulty == 'hard'

    arrival_points, finish_points, road_dir, junction = get_add_mat(dims, grid, difficulty)

    n_turn1 = 3 # 0 - straight, 1-right, 2-left
    n_turn2 = 1 if difficulty == 'medium' else 3


    routes=[]
    # routes for each arrival point
    for i in range(len(arrival_points)):
        paths = []
        # turn 1
        for turn_1 in range(n_turn1):
            # turn 2
            for turn_2 in range(n_turn2):
                total_turns = 0
                curr_turn = turn_1
                path = []
                visited = set()
                current = arrival_points[i]
                path.append(current)
                start = current
                turn_step = 0
                # "start"
                while not goal_reached(i, current, finish_points):
                    visited.add(current)
                    current, turn_prog, turn_completed = next_move(current, curr_turn, turn_step, start, grid, road_dir, junction, visited)
                    if curr_turn == 2 and turn_prog:
                        turn_step +=1
                    if turn_completed:
                        total_turns += 1
                        curr_turn = turn_2
                        turn_step = 0
                        start = current
                    # keep going straight till the exit if 2 turns made already.
                    if total_turns == 2:
                        curr_turn = 0
                    path.append(current)
                paths.append(path)
                # early stopping, if first turn leads to exit
                if total_turns == 1:
                    break
        routes.append(paths)
    return routes

"""
I gotta give credit to most of the Node class and search function to someone on StackExchange. It was just a lot cleaner
that what I orignally had. And the search function is much faster.
"""
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt

import random
import heapq
import numpy as np
import copy


class NotFoundError(Exception): pass


class Node:
    valid_move = {
        'U': lambda a: a[0] > 0,
        'D': lambda a: a[0] < 4,
        'L': lambda a: a[1] > 0,
        'R': lambda a: a[1] < 4
    }

    def __init__(self, width, height, matrix=None, blank=None, parent=None, target=None, action=None):
        assert(width > 1 and height > 1)
        self.width = width
        self.height = height
        self.cells = width * height
        if matrix is None:
            matrix = tuple(range(self.cells))
            blank = self.cells - 1
        assert(len(matrix) == self.cells)
        assert(0 <= blank < self.cells)
        self.matrix = matrix
        self.blank = blank
        self.parent = parent
        self.target = target
        self.action = action

    def __repr__(self):
        return 'Node({0.width}, {0.height}, {0.matrix})'.format(self)

    def __lt__(self, other):
         return self.heuristic() < other.heuristic()

    def __eq__(self, other):
        return self.matrix == other.matrix

    def __ne__(self, other):
        return self.matrix != other.matrix

    def __hash__(self):
        return hash(self.matrix)

    def shuffled(self, swaps=20):
        """Return a new position after making 'swaps' swaps."""
        result = self
        for _ in range(swaps):
            result = random.choice(list(result.neighbors()))
        return result

    def heuristic(self):
        """ This is a bunch custom "heuristics" to solve the puzzle """
        total = 0
        if type(self.target) == tuple:
            # new_heuristic = lambda x: man_dist(x, x.target[0], x.target[1])
            total += man_dist(self, self.target[0], self.target[1])

            # The distance between the two target tiles and blank
            # Keep working on the goal
            total += man_dist(self, self.target[0], 0)
            total += man_dist(self, self.target[1], 0)

            # The distance between the target tiles and goal positions
            total += man_dist(self, self.target[0], self.matrix[self.target[0] - 1])
            total += man_dist(self, self.target[1], self.matrix[self.target[1] - 1])
        else:
            # costs more to move the blank tile away from the target
            total += man_dist(self, self.target, 0)

            # manhattan distance from the target tile to its correct location
            total += man_dist(self, self.target, self.matrix[self.target-1])

            # Manhattan distance between blank and goal location - not sure if this helps
            # total += man_dist(self, 0, self.matrix[self.target - 1])
        return total

    def swapped(self, c, action=None):
        """Return a new position with cell 'c' swapped with the blank."""
        assert(c != self.blank)
        i, j = sorted([c, self.blank])
        return Node(width=self.width, height=self.height, matrix=
                        self.matrix[:i] + (self.matrix[j],)
                        + self.matrix[i+1:j] + (self.matrix[i],)
                        + self.matrix[j+1:], blank=c, parent=self, target=self.target, action=action)

    def neighbors(self):
        """Generate the neighbors to this position, namely the positions
        reachable from this position via a single swap.
        """
        zy, zx = divmod(self.blank, self.width)
        if zx > 0:
            yield self.swapped(self.blank - 1, 'L')
        if zx < self.width - 1:
            yield self.swapped(self.blank + 1, 'R')
        if zy > 0:
            yield self.swapped(self.blank - self.width, 'U')
        if zy < self.height - 1:
            yield self.swapped(self.blank + self.width, 'D')

    def is_valid(self, action):
        """ Check is a move is valid """
        if self.valid_move.get(action):
            return self.valid_move[action](divmod(self.blank, self.width))
        else:
            return False

    def move(self, action):
        matrix = self._move(action)
        if matrix is None:
            return False
        else:
            self.matrix = matrix
            self.blank = self.matrix.index(0)
            return True

    def _move(self, action):
        """ Perform a move on the node - meant for manually altering the state """
        if not self.is_valid(action):
            print('{}: not valid.'.format(action))
            return self.matrix
        zy, zx = divmod(self.blank, self.width)
        if action == 'L':
            if zx > 0:
                return self.do_move(self.blank - 1)
        if action == 'R':
            if zx < self.width - 1:
                return self.do_move(self.blank + 1)
        if action == 'U':
            if zy > 0:
                return self.do_move(self.blank - self.width)
        if action == 'D':
            if zy < self.height - 1:
                return self.do_move(self.blank + self.width)

    def do_move(self, c):
        assert (c != self.blank)
        i, j = sorted([c, self.blank])
        return self.matrix[:i] + (self.matrix[j],) + self.matrix[i + 1:j] + (self.matrix[i],) + self.matrix[j + 1:]


def get_search_stuff():
    """ Return tuple of targets, constraints, subgoals """
    # List of targets for each subgoal
    targets = [1, 2, 3, (4,5), (4,5),
               6, 7, 8, (9, 10), (9, 10),
               11, 12, 13, (14, 15), (14, 15),
               (16, 21), (17, 22), (18, 23), (19, 24), 20]

    # As the puzzle progresses, constrain the actions
    constraints = [
        [],
        [(0, 0)],
        [(0, 0), (0, 1)],
        [(0, 0), (0, 1), (0, 2)],
        [(0, 0), (0, 1), (0, 2),
         (1, 0), (1, 1),
         (2, 0), (2, 1),
         (3, 0), (3, 1),
         (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)],

        [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
        [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0)],
        [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1),
         (3, 2)],
        [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4)],
        [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (2, 0)],
        [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (2, 0), (2, 1)],
        [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (2, 0), (2, 1), (2, 2)],
        [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (2, 0), (2, 1), (2, 2), (3, 0),
         (3, 1), (3, 2)],
        [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (2, 0), (2, 1), (2, 2), (2, 3),
         (2, 4)],
        [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (2, 0), (2, 1), (2, 2), (2, 3),
         (2, 4), (3, 0), (4, 0)],
        [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (2, 0), (2, 1), (2, 2), (2, 3),
         (2, 4), (3, 0), (3, 1), (4, 0), (4, 1)],
        [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (2, 0), (2, 1), (2, 2), (2, 3),
         (2, 4), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2)],
        [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (2, 0), (2, 1), (2, 2), (2, 2),
         (2, 3), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2)]
    ]
    corner_goal = lambda x, t1, t2 : \
                    (x.matrix.index(x.target[0]) == t1 and (x.matrix.index(x.target[1]) == t2))\
                    or (x.matrix[x.target[0] - 1:x.target[1] - 1] == x.target)
    subgoals = [
        lambda x: x.matrix[x.target - 1] == x.target,
        lambda x: x.matrix[x.target - 1] == x.target,
        lambda x: x.matrix[x.target - 1] == x.target,
        lambda x: corner_goal(x, 9, 14) or corner_goal(x, 8, 13),
        lambda x: x.matrix[x.target[0] - 1:x.target[1]] == x.target,
        lambda x: x.matrix[x.target-1] == x.target,
        lambda x: x.matrix[x.target-1] == x.target,
        lambda x: x.matrix[x.target-1] == x.target,
        lambda x: corner_goal(x, 14, 19),
        lambda x: x.matrix[x.target[0] - 1:x.target[1]] == x.target,
        lambda x: x.matrix[x.target-1] == x.target,
        lambda x: x.matrix[x.target-1] == x.target,
        lambda x: x.matrix[x.target-1] == x.target,
        lambda x: corner_goal(x, 19, 24),
        lambda x: x.matrix[x.target[0] - 1:x.target[1]] == x.target,
        lambda x: corner_goal(x, 15, 20),
        lambda x: corner_goal(x, 16, 21),
        lambda x: corner_goal(x, 17, 22),
        lambda x: corner_goal(x, 18, 23),
        lambda x: x.matrix[x.target-1] == x.target
    ]
    return targets, constraints, subgoals

def man_dist(node, a=None, b=None):
    """ Get the l2 distance between 2 points. If a,b=None, return the sum of all entries """
    if a is None and b is None:
        count = 0
        for i, j in enumerate(node.matrix):
            if i == 0:
                count += man_dist(node, 0, node.matrix[-1])
            count += man_dist(node, i, node.matrix[i-1])
        return count
    else:
        ax, ay = divmod(node.matrix.index(a), node.width)
        bx, by = divmod(node.matrix.index(b), node.width)
        return abs(ax-bx) + abs(ay - by)


def is_solvable(state, goal_state, size):

    def count_inversions(state, goal_state, size):
        res = 0
        for i in range(size * size - 1):
            for j in range(i + 1, size * size):
                vi = state[i]
                vj = state[j]
                if goal_state.index(vi) > goal_state.index(vj):
                    res += 1
        return res
    inversions = count_inversions(state, goal_state, size)
    state_blank_row = state.index(0) // size
    state_blank_col = state.index(0) % size
    goal_blank_row = goal_state.index(0) // size
    goal_blank_col = goal_state.index(0) % size
    taxicab = abs(state_blank_row - goal_blank_row) + abs(state_blank_col - goal_blank_col)
    if taxicab % 2 == 0 and inversions % 2 == 0:
        return True
    if taxicab % 2 == 1 and inversions % 2 == 1:
        return True
    return False


def pretty(node):
    """ Pretty printing of the node state """
    print('\n'.join(
        [''.join(['{:4}'.format(item) for item in row])
         for row in np.reshape(node.matrix, (node.height, node.width))]))
    print()


def get_neighbor_costs(node):
    return [(n.action, n.heuristic()) for n in node.neighbors()]


def backtrack(node, parent=None):
    solution_sequence = list()
    solution_sequence.append(node)
    while True:
        node = node.parent
        if node is parent or node is None:
            break
        solution_sequence.append(node)
    solution_sequence.reverse()
    return solution_sequence


def search(start_node, goal_check=None, no_touch=None):
    """
    A* star search
    :param start_node: The inital starting node
    :param goal_check: an anonmyous function to evaluate the goal state
    :param no_touch: List of tuples that represent tile coordinates to disregard
    :return:
    """
    off_limits = set()
    if no_touch:
        for no in no_touch:
            off_limits.add(no)

    explored = set()
    start_data = [start_node.heuristic(), 0, start_node, None]
    frontier = {start_node: start_data}
    open_heap = [start_data]

    while open_heap:
        new_node = heapq.heappop(open_heap)
        node_f, node_g, node, parent_data = new_node
        # pretty(node)
        if goal_check(node):
            # print('Goal reached')
            # pretty(node)
            # print(node)
            return node
        del frontier[node]
        explored.add(node)
        for child in node.neighbors():
            if child in explored:
                continue
            if get_tile_position(child, 0) in off_limits:
                explored.add(child)
                continue

            child_g = node_g
            child_f = child_g + child.heuristic()
            child_data = [child_f, child_g, child, new_node]

            if child not in frontier:
                # pretty(child)
                frontier[child] = child_data
                heapq.heappush(open_heap, child_data)
            else:
                old_neighbor_data = frontier[child]
                if child_data < old_neighbor_data:
                    old_neighbor_data[:] = child_data
                    heapq.heapify(open_heap)
    pretty(start_node)
    raise NotFoundError("No solution for {}".format(start_node))


def move(node, actions, verbose=False):
    """ Pass a string of moves to perform (for manual manipulation of the puzzle)"""
    for action in actions:
        node.move(action)
        pretty(node)
        if verbose:
            print('h={}'.format(node.heuristic()))


def get_tile_position(node, target):
    """ Return (row,col) of a tile """
    return divmod(node.matrix.index(target), node.width)


def get_goal_position(node, target):
    """ Get (row,col) of where a tile belongs. e.g tile 1 -> (0,0) """
    return divmod(target-1, node.width)


def parse_file(file):
    """ Parse starting state from file """
    with open(file, 'r') as f:
        data =f.read()
    data = ' '.join(data.replace('|', ' ').split())
    data = tuple(np.fromstring(data.replace('X', '0'), dtype='int', sep=' ').reshape(25))
    return data


def solve(node, targets, constraints, subgoals):
    count = 0
    count_list = [0]

    estimate = [man_dist(node)]

    for target in targets:
        print('Target tile(s): {}'.format(target))
        print('Initial state:')
        pretty(node)
        node.target = target
        old_node = copy.deepcopy(node)
        leave_out = constraints.pop(0)
        blank_location = get_tile_position(node, 0)

        # Blank tile might be in border of tiles that are being left out.
        if blank_location in leave_out:
            # If so, remove the blank from leave_out
            leave_out.remove(blank_location)
        node = search(node, goal_check=subgoals.pop(0), no_touch=leave_out)
        path = backtrack(node, parent=old_node)
        count_list.append(len(backtrack(node, parent=old_node)))
        estimate.append(man_dist(node))
        print('Estimate: {}'.format(estimate[-1]))
        print('Number of moves: {}'.format(count_list[-1] - count_list[-2]))

    goal_node = backtrack(node)
    # print(count_list)
    # print(estimate)
    return goal_node, estimate, count_list


def make_random_state(goal):
    state = list(goal)
    while True:
        random.shuffle(state)
        if is_solvable(state, goal, 5):
            break
    return tuple(state)


def get_action_sequence(path):
    action_seq = path[1].action
    for sol in path[2:]:
        try:
            action_seq += ',' + sol.action
        except TypeError:
            pass
    return action_seq

def run_one(node):
    targets, constraints, subgoals = get_search_stuff()
    solution, _, _ = solve(node, targets=targets, constraints=constraints, subgoals=subgoals)
    print('Total number of moves: {}'.format(len(solution)))


def run_multiple(nruns=1, plotting=False):
    heuristic_estimates = []
    move_counts = []
    for i in range(nruns):
        targets, constraints, subgoals = get_search_stuff()
        state = make_random_state(goal)
        node = Node(width=5, height=5, matrix=state, blank=state.index(0), target=1)
        solution, est, counts = solve(node,  targets=targets, constraints=constraints, subgoals=subgoals)
        print('Total number of moves: {}'.format(len(solution)))
        heuristic_estimates.append(est)
        move_counts.append(counts)

    if plotting:
        plt.figure()
        for e in heuristic_estimates:
            plt.plot(e)
            plt.xlim([0, 20])
            plt.ylim([0, 120])
        plt.figure()
        for c in move_counts:
            plt.plot(np.diff(c))
            plt.xlim([0, 20])
            plt.ylim([0, 200])
        plt.figure()
        for c in move_counts:
            plt.plot(c)
            plt.xlim([0, 20])
            plt.ylim([0, 600])
        plt.show()


goal = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 0)
# state = (0, 6, 23, 22, 21, 9, 13, 2, 11, 8, 4, 5, 7, 19, 15, 16, 10, 12, 3, 20, 17, 18, 24, 14, 1)

# Load a state from file
# state = parse_file('init.txt')

state = make_random_state(goal)
# Create the initial node with goal target of 1

initial = Node(width=5, height=5, matrix=state, blank=state.index(0), target=1)
run_one(initial)

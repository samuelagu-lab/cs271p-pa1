"""
CS 271P - PA 1,2B: Movement Range
==============================
SUBMISSION FILE. Submit ONLY this file, named exactly `reachable_targets.py`.

Implement reachable_targets(...) below. See the Instructions PDF
("PA 1,2B - Movement Range - Submission Instructions.pdf") for the full problem
statement, the rules, the restrictions, and how grading works. Test your work
locally first by running:  python3 test_local.py

PROVIDED TO YOUR FUNCTION (do NOT import or redefine these):
  - the `grid` object, with:
        grid.in_bounds(t)        -> True if tile t = (x, y) is on the board
        grid.king_neighbors(t)   -> the up-to-8 in-bounds king-move neighbors of t
        grid.W, grid.H           -> board width (columns) and height (rows)
        grid.obstacles           -> the set of wall tiles
  - cheb(p, q)                   -> king-move (Chebyshev) distance between two tiles

Tiles are (x, y) with x = column, y = row (row 0 at the TOP).
You may use the Python 3 standard library (e.g. `from math import ceil`,
`from collections import deque`). Do NOT use third-party packages.
"""
import math
from collections import deque


def BFS_GRID(state, grid, r, b):
    """Performs BFS on grid given initial state(s). 

    Args:
        state (set or int): the start state(s) of our search
        grid (_type_): the data structure we will perform the search
        r (int): radius from our initial state we can explore
        b (tuple): Bs location on the grid

    Returns:
        set : returns set of tiles explored
    """
    if isinstance(state, set):
        reached = {t: 0 for t in state}
        frontier = deque(state)
    else:  # single (x, y) tile
        reached = {state: 0}
        frontier = deque([state])
    blocked = set(grid.obstacles) | {b}
    
    # BFS search begins
    while frontier:
        node = frontier.popleft()
        d = reached[node]
        if d == r:
            continue
        nd = d + 1
        for n in grid.king_neighbors(node):
            if n in blocked or n in reached:
                continue
            reached[n] = nd
            frontier.append(n)

    return reached


def reachable_targets(grid, A, B, base_movement, path):
    """Return (next_waypoints, adjacent_reachable) as two sets of (x, y) tiles.

    Rules (see the Instructions PDF for the full statement):
      - Movement is 8-directional (king moves), cost 1 each. B's tile and every
        tile in grid.obstacles are BLOCKED (cannot be entered or passed through).
      - M = ceil(base_movement / 2). The path EXCLUDES A, so it costs
        T = len(path) moves; the remaining budget is R = M - T, measured from the
        path end e = path[-1] (or e = A if path is empty).
      - next_waypoints: every passable tile w you could reach from e AND from
        which a tile adjacent to B is still reachable, all within R moves total.
      - adjacent_reachable: every passable tile adjacent to B reachable directly
        from e within R moves.

    Provided: grid.in_bounds(t), grid.king_neighbors(t), grid.W, grid.H,
    grid.obstacles, and cheb(p, q).
    """
    # Task: build and return the two sets defined above —
    #   next_waypoints    : tiles you could set as the next waypoint
    #   adjacent_reachable : B-adjacent tiles reachable directly from the path end
    # Available: the provided primitives (grid.in_bounds, grid.king_neighbors, grid.W, grid.H,
    #            grid.obstacles, cheb) — see the docstrings above.

    next_waypoints = set()
    adjacent_reachable = set()

    # --------------------------------------- Set up
    # Remaining budget: half of base movement
    M = math.ceil(base_movement / 2)
    T = len(path)
    R = M - T
    if R < 0:
        return next_waypoints, adjacent_reachable

    # if path empty then we start at A, else start at end of path
    if len(path) == 0:
        initial_state = A
    else:
        initial_state = path[-1]

    # store possible goal states: the ring of passable tiles adjacent to B.
    goal_states = {t for t in grid.king_neighbors(B) if t not in grid.obstacles}
    if not goal_states:
        return next_waypoints, adjacent_reachable

    # --------------------------------------- Begin Search
    states_from_e = BFS_GRID(
        initial_state,
        grid,
        R,
        B
    )

    # adjacent_reachable: states within R moves of the path end/initial state.
    adjacent_reachable = {state for state in goal_states if state in states_from_e}

    # if no ring tile is reachable from e, no waypoint can exist
    if not adjacent_reachable:
        return next_waypoints, adjacent_reachable

    states_from_ring = BFS_GRID(
        goal_states,
        grid,
        R,
        B
    )

    # waypoints: state w with dist_e[w] + dist_ring[w] <= R.
    for state, d_e in states_from_e.items():
        if d_e + states_from_ring.get(state, float('inf')) <= R:
            next_waypoints.add(state)

    return next_waypoints, adjacent_reachable

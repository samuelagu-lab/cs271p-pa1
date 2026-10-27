#!/usr/bin/env python3
"""
CS 271P - PA 1,2B: Movement Range  --  LOCAL SELF-TEST
===================================================
Run from this folder:   python3 test_local.py
                        python3 test_local.py --final      (a longer test, see below)

It checks your reachable_targets.py for:
  (1) CORRECTNESS  -- on fixed edge-case boards plus many random boards.
  (2) SPEED        -- whether it runs faster than a deliberately naive baseline.

WHAT A PASS MEANS (read this): a PASS guarantees only that you are in roughly the
70%+ range on Gradescope. It does NOT guarantee full marks -- your exact score
(up to a small 105% bonus) depends on how fast your solution is, graded on
Gradescope against reference solutions that are NOT shown here, with a buffer
applied in your favor for timing and coding-style variation.

THE --final OPTION. Your grade comes from a FINAL grading run after the deadline. That run
uses many more boards, and it also times a fast solution on much larger boards, about 160 by
160 tiles. "python3 test_local.py --final" imitates it here:
  * 500 random correctness boards, where the plain test uses 40. The fixed edge-case boards
    stay the same.
  * a timing of your solution on 40 random boards of 150 to 170 tiles.
  * NEW random boards on every run. The seed is printed, and "--seed N" repeats a run.
The large boards are only timed. Their answers are not checked here, because the naive
baseline is far too slow to compute them. A solution that scales well needs well under one
second per large board. This test cannot tell you your speed tier. It tells you whether the
large boards are a problem for your solution.

This file is yours to read and run. It contains only the naive baseline -- not
the reference solutions used for the higher speed tiers.

Requires only the Python 3 standard library.
"""
from dataclasses import dataclass, field
from math import ceil
import argparse
import importlib
import random
import signal
import time
import sys


# ============================================================ board model
# (Identical to the primitives your function is given when graded.)
def cheb(p, q):
    "King-move (Chebyshev) distance between two tiles."
    return max(abs(p[0] - q[0]), abs(p[1] - q[1]))


@dataclass
class Grid:
    W: int
    H: int
    obstacles: frozenset = field(default_factory=frozenset)

    def in_bounds(self, t):
        return 0 <= t[0] < self.W and 0 <= t[1] < self.H

    def king_neighbors(self, t):
        "The up-to-8 in-bounds king-move neighbors of t (ignores obstacles/B)."
        x, y = t
        out = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                n = (x + dx, y + dy)
                if self.in_bounds(n):
                    out.append(n)
        return out


@dataclass
class Scenario:
    grid: Grid
    A: tuple
    B: tuple
    base_movement: int
    path: list = field(default_factory=list)


def parse_board(ascii_map):
    "Build (grid, A, B) from text. '.'=open  '#'=wall  'A'=A  'B'=B."
    lines = ascii_map.strip("\n").split("\n")
    H = len(lines)
    W = max(len(ln) for ln in lines)
    obstacles = set()
    A = B = None
    for y, ln in enumerate(lines):
        for x, ch in enumerate(ln):
            if ch == "#":
                obstacles.add((x, y))
            elif ch == "A":
                A = (x, y)
            elif ch == "B":
                B = (x, y)
    return Grid(W, H, frozenset(obstacles)), A, B


def validate(scn):
    "Return a list of setup-error strings (empty list == OK)."
    g, A, B, path = scn.grid, scn.A, scn.B, scn.path
    errs = []
    if A is None or not g.in_bounds(A):
        errs.append("A is missing / out of bounds")
    if B is None or not g.in_bounds(B):
        errs.append("B is missing / out of bounds")
    if A == B:
        errs.append("A and B are on the same tile")
    if A in g.obstacles:
        errs.append("A is on an obstacle")
    if B in g.obstacles:
        errs.append("B is on an obstacle")
    M = ceil(scn.base_movement / 2)
    if len(path) > M:
        errs.append("path is too long for the budget")
    prev = A
    for i, t in enumerate(path):
        if not g.in_bounds(t):
            errs.append(f"path[{i}] {t} is out of bounds")
        elif t in g.obstacles:
            errs.append(f"path[{i}] {t} is on an obstacle")
        if t == B:
            errs.append(f"path[{i}] {t} is B's tile")
        if prev is not None and cheb(prev, t) != 1:
            errs.append(f"path[{i}] {t} is not king-adjacent to the previous tile")
        prev = t
    return errs


# ====================================== naive reference (the visible baseline)
# Deliberately the most costly, most naive method: it floods OUT from EVERY tile
# on the board, one at a time, sharing nothing between tiles. It is the
# correctness gold standard AND the speed baseline you must beat.
def _ring_tiles(scn):
    "Tiles adjacent to B (part of the problem statement, not a search)."
    blocked = set(scn.grid.obstacles) | {scn.B}
    return [t for t in scn.grid.king_neighbors(scn.B) if t not in blocked]


def _flood(scn, start, kmax):
    "Naive flood: {tile: min king-moves from start} out to radius kmax (B & walls blocked)."
    blocked = set(scn.grid.obstacles) | {scn.B}
    if not scn.grid.in_bounds(start) or start in blocked:
        return {}
    dist = {start: 0}
    frontier = [start]
    for k in range(1, kmax + 1):
        nxt = []
        for u in frontier:
            for v in scn.grid.king_neighbors(u):
                if v not in blocked and v not in dist:
                    dist[v] = k
                    nxt.append(v)
        frontier = nxt
        if not frontier:
            break
    return dist


def naive_reachable_targets(grid, A, B, base_movement, path):
    "Naive brute force: check EVERY tile one at a time, each with its OWN floods."
    scn = Scenario(grid, A, B, base_movement, list(path))
    M = ceil(base_movement / 2)
    R = M - len(path)
    if R < 0:
        return set(), set()
    e = path[-1] if path else A
    blocked = set(grid.obstacles) | {B}
    ring = set(_ring_tiles(scn))
    INF = float("inf")
    waypoints, adjacent = set(), set()
    for x in range(grid.W):
        for y in range(grid.H):
            w = (x, y)
            if w in blocked:
                continue
            dist = _flood(scn, w, R)                                  # flood OUT from this tile
            d_to_e = dist.get(e, INF)                                 # distance to the path end
            d_to_ring = min((dist.get(t, INF) for t in ring), default=INF)  # to nearest B-adjacent tile
            if d_to_e + d_to_ring <= R:                               # both legs fit the budget?
                waypoints.add(w)
            if w in ring and d_to_e <= R:                             # B-adjacent tile reachable from e
                adjacent.add(w)
    return waypoints, adjacent


# ============================================================ test batteries
def _scn(ascii_map, base, path):
    g, A, B = parse_board(ascii_map)
    return Scenario(g, A, B, base, path)


# Fixed edge cases (same family as the in-notebook suite).
SUITE = {
    "open / empty path": _scn(
        "..........\n..........\n..A....B..\n..........\n..........\n", 10, []),
    "ends adjacent to B": _scn(
        "...........\n...........\n..A.....B..\n...........\n...........\n",
        12, [(3, 2), (4, 2), (5, 2), (6, 2), (7, 2)]),
    "path with a repeat": _scn(
        ".........\n.........\n..A...B..\n.........\n.........\n",
        12, [(3, 2), (3, 3), (3, 2)]),
    "R = 0 (full budget used)": _scn(
        ".........\n..A...B..\n.........\n", 6, [(3, 1), (4, 1), (5, 1)]),
    "wall forces a detour": _scn(
        ".............\n....#####....\n.A..#...#..B.\n....#...#....\n....#####....\n.............\n",
        18, []),
    "obstacle on B's ring": _scn(
        "...........\n......#....\n..A...B....\n...........\n...........\n", 8, []),
    "B in the corner": _scn(
        "..........\n..........\n.....A....\n.........B\n", 8, []),
    "odd base (ceil)": _scn("........\n..A..B..\n........\n", 5, []),
}


def _random_scenario(rng):
    "A random but VALID scenario (passable A/B, a legal king-walk forced path)."
    W = rng.randint(8, 16)
    H = rng.randint(6, 14)
    density = rng.choice([0.0, 0.08, 0.15, 0.22])
    cells = [(x, y) for x in range(W) for y in range(H)]
    obstacles = {c for c in cells if rng.random() < density}
    free = [c for c in cells if c not in obstacles]
    A = rng.choice(free)
    B = rng.choice([c for c in free if c != A])
    grid = Grid(W, H, frozenset(obstacles))
    base = rng.randint(3, 14)
    M = ceil(base / 2)
    blocked = set(obstacles) | {B}
    T = rng.randint(0, M)
    path = []
    prev = A
    for _ in range(T):
        nbrs = [n for n in grid.king_neighbors(prev) if n not in blocked]
        if not nbrs:
            break
        prev = rng.choice(nbrs)
        path.append(prev)
    return Scenario(grid, A, B, base, path)


# A few medium open boards used only for timing (naive does real work here).
def _open(W, H, A, B, base):
    return Scenario(Grid(W, H, frozenset()), A, B, base, [])


TIMING = [
    _open(18, 18, (3, 9), (13, 9), 22),
    _open(20, 20, (4, 10), (15, 10), 24),
    _open(22, 22, (4, 11), (16, 11), 26),
]


# ---- used only by --final ------------------------------------------------------------------
FINAL_RANDOM_BOARDS = 500        # random correctness boards (the plain test uses 40)
LARGE_BOARDS = 40                # random large boards, timed only
LARGE_SIZE = (150, 170)          # tiles per side, as in the final grading run
LARGE_TIME_LIMIT = 5.0           # seconds for one board, the grader's own limit
PROBE_SIZES = [(40, 44), (80, 88)]   # two smaller boards, timed first as a guard
PROBE_TOO_SLOW = 1.5             # seconds. Slower than this, and a 160-tile board would time out


def _big_scenario(rng, lo, hi):
    """A random open-ish board of lo to hi tiles per side, with an empty forced path and a budget
    that reaches most of the way across it. For timing only."""
    W, H = rng.randint(lo, hi), rng.randint(lo, hi)
    density = rng.choice([0.0, 0.05, 0.10])
    cells = [(x, y) for x in range(W) for y in range(H)]
    obstacles = {c for c in cells if rng.random() < density}
    free = [c for c in cells if c not in obstacles]
    base = rng.randint(lo - 10, hi - 10)
    M = ceil(base / 2)
    A = rng.choice(free)
    B = None
    for _ in range(400):                       # B within reach of A, and not trivially close
        cand = rng.choice(free)
        if cand != A and M // 2 <= cheb(A, cand) <= M:
            B = cand
            break
    if B is None:
        B = rng.choice([c for c in free if c != A])
    return Scenario(Grid(W, H, frozenset(obstacles)), A, B, base, [])


class _TooSlow(Exception):
    pass


def _raise_too_slow(signum, frame):
    raise _TooSlow()


def _timed(fn, scn, limit):
    """Seconds one call takes, or None when it passes `limit`. Where the system has an alarm
    signal (Linux, macOS) the call is stopped at the limit. On Windows it runs to its end."""
    args = (scn.grid, scn.A, scn.B, scn.base_movement, list(scn.path))
    can_stop = hasattr(signal, "SIGALRM")
    if can_stop:
        signal.signal(signal.SIGALRM, _raise_too_slow)
        signal.setitimer(signal.ITIMER_REAL, limit)
    t0 = time.perf_counter()
    try:
        fn(*args)
    except _TooSlow:
        return None
    finally:
        if can_stop:
            signal.setitimer(signal.ITIMER_REAL, 0)
    dt = time.perf_counter() - t0
    return None if dt > limit else dt


def _large_board_timing(student, rng):
    "The --final part (3): time the student's solution on large random boards. Returns a verdict."
    print("\n(3) Large boards ........ timing only, answers are not checked here")
    for lo, hi in PROBE_SIZES:
        dt = _timed(student, _big_scenario(rng, lo, hi), LARGE_TIME_LIMIT)
        if dt is None or dt > PROBE_TOO_SLOW:
            took = f"more than {LARGE_TIME_LIMIT:g} s" if dt is None else f"{dt:.1f} s"
            print(f"      A board of about {lo} by {lo} tiles took {took}. The large boards are\n"
                  f"      about {LARGE_SIZE[0]} by {LARGE_SIZE[0]}, about {round((LARGE_SIZE[0] / lo) ** 2)} "
                  f"times the area, so they were not run.\n"
                  f"      In the final grading run a solution this slow stays below 95%.")
            return False
    times, timeouts = [], 0
    for i in range(LARGE_BOARDS):
        try:
            dt = _timed(student, _big_scenario(rng, *LARGE_SIZE), LARGE_TIME_LIMIT)
        except Exception as ex:  # noqa: BLE001
            print(f"      Your function raised an error on large board {i + 1}: {ex!r}\n"
                  f"      In the final grading run that removes all speed credit.")
            return False
        if dt is None:
            timeouts += 1
            if timeouts >= 3:
                print(f"      3 large boards took more than {LARGE_TIME_LIMIT:g} s each, so the rest "
                      f"were not run.")
                return False
        else:
            times.append(dt)
    avg = sum(times) / len(times)
    print(f"      {len(times)} boards of {LARGE_SIZE[0]} to {LARGE_SIZE[1]} tiles per side: "
          f"average {avg * 1e3:.0f} ms, slowest {max(times) * 1e3:.0f} ms"
          + (f", and {timeouts} over the {LARGE_TIME_LIMIT:g} s limit" if timeouts else ""))
    ok = avg < 1.0 and not timeouts
    print("      " + ("No board came near the time limit. Whether this is fast enough for 95%, 100% or\n"
                      "      105% is decided on Gradescope, against reference solutions not shown here.\n"
                      "      A lower time is better."
                      if ok else
                      "A solution that scales well needs well under one second per board. Expect to\n"
                      "      lose speed tiers on these boards in the final grading run."))
    return ok


# ============================================================ helpers
def _ascii_diff(scn, expected, student):
    "Render the board: o=correct  m=missed  x=wrong  #=wall  A B  .=other."
    g = scn.grid
    ok = expected & student
    missed = expected - student
    wrong = student - expected
    rows = []
    for y in range(g.H):
        row = []
        for x in range(g.W):
            t = (x, y)
            if t in g.obstacles:
                ch = "#"
            elif t == scn.A:
                ch = "A"
            elif t == scn.B:
                ch = "B"
            elif t in missed:
                ch = "m"
            elif t in wrong:
                ch = "x"
            elif t in ok:
                ch = "o"
            else:
                ch = "."
            row.append(ch)
        rows.append("".join(row))
    return "\n".join(rows)


def _best_time(fn, args, reps=3):
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args)
        dt = time.perf_counter() - t0
        if dt < best:
            best = dt
    return best


def _load_student():
    try:
        mod = importlib.import_module("reachable_targets")
    except Exception as ex:  # noqa: BLE001
        print("ERROR: could not import reachable_targets.py:", repr(ex))
        print("Make sure reachable_targets.py is in this folder and has no syntax errors.")
        sys.exit(1)
    if not hasattr(mod, "reachable_targets"):
        print("ERROR: reachable_targets.py must define a function named reachable_targets(...).")
        sys.exit(1)
    mod.cheb = cheb  # provide the primitive the way the grader does
    return mod.reachable_targets


# ============================================================ main
def main():
    ap = argparse.ArgumentParser(description="Local self-test for PA 1,2B: Movement Range.")
    ap.add_argument("--final", action="store_true",
                    help="a longer test like the final grading run: more random boards, and a "
                         "timing on large boards. New random boards on every run")
    ap.add_argument("--seed", type=int, help="with --final, repeat the run that printed this seed")
    opts = ap.parse_args()
    if opts.seed is not None and not opts.final:
        ap.error("--seed goes with --final. The plain test always uses the same boards.")

    student = _load_student()

    # The plain test uses the same 40 random boards every time. --final draws new ones.
    seed = 171 if not opts.final else opts.seed if opts.seed is not None \
        else random.SystemRandom().randrange(1, 10 ** 9)
    rng = random.Random(seed)
    battery = list(SUITE.items())
    for i in range(FINAL_RANDOM_BOARDS if opts.final else 40):
        scn = _random_scenario(rng)
        if validate(scn):
            continue
        battery.append((f"random #{i}", scn))

    # ---- (1) correctness
    fails = []
    for name, scn in battery:
        gold = naive_reachable_targets(scn.grid, scn.A, scn.B, scn.base_movement, list(scn.path))
        try:
            sw, sa = student(scn.grid, scn.A, scn.B, scn.base_movement, list(scn.path))
            sw = set(map(tuple, sw))
            sa = set(map(tuple, sa))
        except Exception as ex:  # noqa: BLE001
            fails.append((name, scn, ("raised", repr(ex))))
            continue
        if (sw, sa) != gold:
            fails.append((name, scn, ("wrong", (sw, sa, gold[0], gold[1]))))

    n_total = len(battery)
    correctness_pass = not fails

    print("=" * 64)
    print(" CS 271P - PA 1,2B: Movement Range  --  local self-test")
    print("=" * 64)
    if opts.final:
        print(f"--final: new random boards on this run, seed {seed}.\n"
              f"         To repeat exactly this run:  python3 test_local.py --final --seed {seed}")
    if correctness_pass:
        print(f"(1) Correctness ......... PASS   ({n_total}/{n_total} boards correct)")
    else:
        print(f"(1) Correctness ......... FAIL   ({n_total - len(fails)}/{n_total} boards correct)")
        print("\n    Legend:  o=correct   m=missed (should be in answer)   x=wrong (should not be)")
        for name, scn, info in fails[:2]:
            print(f"\n    --- board '{name}' ---")
            if info[0] == "raised":
                print(f"    your function raised: {info[1]}")
                continue
            sw, sa, tw, ta = info[1]
            print("    next_waypoints:")
            print("      " + _ascii_diff(scn, tw, sw).replace("\n", "\n      "))
            if sa != ta:
                print("    adjacent_reachable:")
                print("      " + _ascii_diff(scn, ta, sa).replace("\n", "\n      "))
        if len(fails) > 2:
            print(f"\n    ... and {len(fails) - 2} more failing board(s).")

    # ---- (2) speed vs the naive baseline (only meaningful once correct)
    if not correctness_pass:
        print("\n(2) Speed vs naive ...... (skipped -- fix correctness first)")
        speed_pass = False
    else:
        s_tot = n_tot = 0.0
        for scn in TIMING:
            args = (scn.grid, scn.A, scn.B, scn.base_movement, list(scn.path))
            s_tot += _best_time(student, args, 3)
            n_tot += _best_time(naive_reachable_targets, args, 3)
        speed_pass = s_tot < n_tot
        verdict = "PASS   (faster than the naive baseline)" if speed_pass \
            else "FAIL   (not faster than the naive baseline)"
        print(f"(2) Speed vs naive ...... {verdict}")
        print(f"      your solution: {s_tot * 1e3:7.1f} ms total    naive baseline: {n_tot * 1e3:7.1f} ms total")

    # ---- (3) --final only: large boards, once correct and faster than naive
    large_ok = None
    if opts.final and correctness_pass and speed_pass:
        large_ok = _large_board_timing(student, rng)
    elif opts.final:
        print("\n(3) Large boards ........ (skipped -- pass (1) and (2) first)")

    overall = correctness_pass and speed_pass
    print("-" * 64)
    print(f" OVERALL: {'PASS' if overall else 'FAIL'}"
          + (", but read part (3) about the large boards" if overall and large_ok is False else ""))
    print("-" * 64)

    if overall:
        print(
            "What this PASS means:\n"
            "  Your solution is correct AND faster than the naive baseline. On\n"
            "  Gradescope that guarantees only that you are in roughly the 70%+\n"
            "  range. It does NOT guarantee full marks: your exact score (up to a\n"
            "  small 105% bonus) depends on how fast your solution is, graded on\n"
            "  Gradescope against reference solutions not shown here, with a buffer\n"
            "  applied in your favor for timing and coding-style variation."
        )
    elif correctness_pass:
        print(
            "You are CORRECT but not faster than the naive baseline. A correct\n"
            "solution still earns at least the 70% floor on Gradescope -- but you\n"
            "should be able to do much better than flooding from every tile. See\n"
            "the hint in the Instructions PDF, then re-run this test."
        )
    else:
        print(
            "Fix correctness first. A correct solution earns at least the 70%\n"
            "floor on Gradescope; speed then determines the rest."
        )


if __name__ == "__main__":
    main()

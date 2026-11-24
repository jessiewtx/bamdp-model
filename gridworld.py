from collections import deque
import numpy as np


TRAINING = [
    ".........",
    "@.G.....$",
    ".........",
]

TRANSFER = [
    "..G......",
    ".........",
    "@.......$",
]

def parse(rows):
    """
    Turns a list of strings into dict of (x, y) -> character.
    """

    height = len(rows)
    width = len(rows[0])
    tiles = {}

    start = goal = green = None

    for x in range(width):

        for y in range(height):

            char = rows[height - 1 - y][x]
            tiles[(x, y)] = char

            if char == "@":
                start = (x, y)

            elif char == "$":
                goal = (x, y)

            elif char == "G":
                green = (x, y)

    return tiles, start, goal, green


RIGHT, LEFT, UP, DOWN = (1, 0), (-1, 0), (0, 1), (0, -1)
DIRECTIONS = [RIGHT, LEFT, UP, DOWN]


def walkable(tiles, loc):
    """
    True if loc on grid and not wall
    """
    return (loc in tiles) and (tiles[loc] != "#")


def next_neighbors(tiles, state):
    """
    All places for next step
    """
    x, y, _ = state

    for d in DIRECTIONS:

        nxt = (x + d[0], y + d[1])

        if walkable(tiles, nxt):

            yield d, (nxt[0], nxt[1], d)


JUMP_LENGTH = 4 # length of jump

def jump(tiles, state):
    '''
    Jump or none if not possible (like lands on non-valid loc)
    '''

    x, y, facing = state

    if tiles.get((x, y)) != "G":
        return None

    landing = (x + facing[0] * JUMP_LENGTH, y + facing[1] * JUMP_LENGTH)

    if not walkable(tiles, landing):

        return None

    return (landing[0], landing[1], facing)



def at(loc):
    """
    At loc
    """
    return lambda s: (s[0], s[1]) == loc


def at_facing(loc, facing):
    """
    At loc and facing direction
    """
    return lambda s: s == (loc[0], loc[1], facing)


def bfs(tiles, start_state, is_goal):
    """
    Fewest steps from start_state to a goal state

    Output is (num_steps, [directions walked in a list])
    """

    seen = {start_state}

    queue = deque([(start_state, [])])

    while queue:

        state, path = queue.popleft()

        if is_goal(state):

            return len(path), path

        for d, next in next_neighbors(tiles, state):

            if next not in seen:

                seen.add(next)

                queue.append((next, path + [d]))

    raise ValueError("no path is possible")

if __name__ == "__main__":

    # # gridworld
    # for name, rows in [("TRAINING", TRAINING), ("TRANSFER", TRANSFER)]:

    #     tiles, start, goal, green = parse(rows)

    #     print(f"{name}  start {start}  green {green}  goal {goal}")


    # # walk, testcases written by AI
    # tiles, start, goal, green = parse(TRANSFER)
    # state = (start[0], start[1], RIGHT)
    # print("from", state)
    # for d, nxt in next_neighbors(tiles, state):
    #     print("  move", d, "->", nxt)


    # # jump, testcases written by AI
    # tiles, start, goal, green = parse(TRANSFER)
    # print("green tile at", green)
    # print("  facing right:", jump(tiles, (green[0], green[1], RIGHT)))
    # print("  facing up:   ", jump(tiles, (green[0], green[1], UP)))
    # print("  from start:  ", jump(tiles, (start[0], start[1], RIGHT)))



    # bfs, testcases written by AI
    for name, rows in [("TRAINING", TRAINING), ("TRANSFER", TRANSFER)]:
        tiles, start, goal, green = parse(rows)
        start_state = (start[0], start[1], RIGHT)

        direct, _ = bfs(tiles, start_state, at(goal))
        to_green, _ = bfs(tiles, start_state, at_facing(green, RIGHT))
        landing = jump(tiles, (green[0], green[1], RIGHT))
        to_goal, _ = bfs(tiles, landing, at(goal))

        print(f"{name}  direct {direct}  S->G {to_green}  jump 1  "
              f"L->X {to_goal}  via-green {to_green + 1 + to_goal}")

from gridworld import (
    TRAINING, TRANSFER, RIGHT, parse, jump, bfs, at, at_facing
)


# estimate expeced num of steps from jump to goal, made with AI help
# called "theta" later on
def posterior(prior_mean, prior_var, ys, obs_var = 1.0):
    """
    Conjugate Gaussian update, in precision form.

    theta ~ N(prior_mean, prior_var), and each demonstration gives
    an observation y ~ N(theta, obs_var).

    prior_mean = initial guess for expected steps
    prior_var = uncertainty
    ys = observed costs (list)
    obs_var = assumed variance of obs
    """

    precision = 1.0 / prior_var + len(ys) / obs_var

    mean = (prior_mean / prior_var + sum(ys) / obs_var) / precision

    return mean, 1.0 / precision


def plan(direct, to_green, theta):
    """
    Compare two routes, return (choice, predicted cost)

    direct = shortest distance walking from start to goal
    to_green = shortest walking distance from start to green tile
    theta = estimated walking cost from jump to goal (see fcn above)
    """

    use_green = to_green + 1 + theta

    if use_green < direct:

        return "green", use_green

    return "direct", float(direct)


def measure(rows):
    """
    Measure costs, find actual num steps for route
    """

    tiles, start, goal, green = parse(rows) # from gridworld, tiles is grid

    start_state = (start[0], start[1], RIGHT)

    direct, _ = bfs(tiles, start_state, at(goal)) # path is not needed

    to_green, green_path = bfs(tiles, start_state, at_facing(green, RIGHT))

    landing = jump(tiles, (green[0], green[1], RIGHT))

    to_goal, goal_path = bfs(tiles, landing, at(goal)) # starts at landing loc

    return {
        "direct": direct,
        "to_green": to_green,
        "to_goal": to_goal,
        "use_green": to_green + 1 + to_goal,
        "moves": green_path + ["JUMP"] + goal_path,
    }

if __name__ == "__main__":

    # testcases for inference, made by AI
    train = measure(TRAINING)
    test = measure(TRANSFER)

    ys = [train["to_goal"]] * 4
    theta, theta_var = posterior(4.0, 1.0, ys)
    print(f"after 4 demos of y={train['to_goal']}: "
          f"theta ~ N({theta:.2f}, {theta_var:.2f})\n")

    models = [
        ("abstract + learned theta",
         plan(test["direct"], test["to_green"], theta)),
        ("abstract + true cost",
         plan(test["direct"], test["to_green"], test["to_goal"])),
        ("flat planner (BFS)",
         ("direct", float(test["direct"]))),
    ]

    print(f"{'model':<26}{'chooses':<10}{'predicted':>10}{'realized':>10}")
    for name, (choice, predicted) in models:
        realized = test["direct"] if choice == "direct" else test["use_green"]
        print(f"{name:<26}{choice:<10}{predicted:>10.1f}{realized:>10}")

        names = {(1, 0): "right", (-1, 0): "left", (0, 1): "up", (0, -1): "down", "JUMP": "JUMP"}
        print("\nexecuted route on transfer level:")
        print("  " + " -> ".join(names[m] for m in test["moves"]))
        print(f"  {len(test['moves'])} actions, {test['use_green']} time steps "
            f"(direct would have been {test['direct']})")

    print("\ndemos  theta  est. via-green  chooses")
    for n in range(0, 7):
        th, _ = posterior(4.0, 1.0, [train["to_goal"]] * n)
        choice, _ = plan(test["direct"], test["to_green"], th)
        print(f"{n:>5}  {th:>5.2f}  {test['to_green'] + 1 + th:>14.2f}  {choice}")

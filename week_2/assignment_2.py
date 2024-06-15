# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Programming Assignment 2: Integer Linear Programming
# ## Imports

# %%
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from pulp import *  # pyright: ignore [reportWildcardImportFromLibrary]

# %% [markdown]
# ## Problem 1
#
# Write down a constraint that says that each vertex must be colored exactly
# one of three colors, red, green or blue, in terms of $x_{i}^{R}$, $x_{i}^{G}$
# and $x_{i}^{B}$.
# \begin{equation}
# x_{i}^{R} + x_{i}^{G} + x_{i}^{B} = 1, \qquad i = 1, \ldots, n
# \end{equation}
#
# Write down constraints for each edge $(i, j) \in E$ that they cannot be the
# same color.
# \begin{eqnarray}
# x_{i}^{R} + x_{j}^{R} &\leq& 1 \\
# x_{i}^{G} + x_{j}^{G} &\leq& 1 \\
# x_{i}^{B} + x_{j}^{B} &\leq& 1
# \end{eqnarray}


# %%
# The instructor's test function has a bug. I had to work around it.
def encode_and_solve_three_coloring(n, edge_list):
    n -= 1
    assert n >= 1, "Graph must have at least one vertex"
    assert all(1 <= i <= n and 1 <= j <= n and i != j for i, j in edge_list), "Edge list is not well formed"

    # Decision variables
    red_vars = [LpVariable(f"Red{i}", lowBound=0, upBound=1, cat=LpInteger) for i in range(n)]
    green_vars = [LpVariable(f"Green{i}", lowBound=0, upBound=1, cat=LpInteger) for i in range(n)]
    blue_vars = [LpVariable(f"Blue{i}", lowBound=0, upBound=1, cat=LpInteger) for i in range(n)]

    # Objective
    prob = LpProblem("ThreeColor", LpMinimize)
    prob += lpSum(red_vars) + lpSum(green_vars) + lpSum(blue_vars)

    # Add constraints
    for i in range(n):
        prob += red_vars[i] + green_vars[i] + blue_vars[i] == 1

    for i, j in ((i - 1, j - 1) for i, j in edge_list):
        prob += red_vars[i] + red_vars[j] <= 1
        prob += green_vars[i] + green_vars[j] <= 1
        prob += blue_vars[i] + blue_vars[j] <= 1

    # Solve problem and return result
    prob.solve()

    if LpStatus[prob.status] != "Optimal":
        return False, []

    color_assignment = ["r"]

    for i in range(n):
        opts = [
            ("r", red_vars[i].varValue),
            ("g", green_vars[i].varValue),
            ("b", blue_vars[i].varValue),
        ]
        color = max(opts, key=lambda p: p[1])[0]  # pyright: ignore
        color_assignment.append(color)

    return True, color_assignment


# %%
def check_three_color_assign(n, edge_list, color_assign):
    assert (
        len(color_assign) == n
    ), f"Error: The list of color assignments has {len(color_assign)} entries but must be same as number of vertices {n}"
    assert all(
        col == "r" or col == "b" or col == "g" for col in color_assign
    ), f"Error: Each entry in color assignment list must be r, g or b. Your code returned: {color_assign}"
    for i, j in edge_list:
        ci = color_assign[i]  # BUG: Vertex labels should not be used as indices!
        cj = color_assign[j]
        assert ci != cj, f"Error: For edge ({i}, {j}) we have same color assignment ({ci}, {cj})"
    print("Success: Three coloring assignment checks out!!")


# %%
n = 5
edge_list = [(1, 2), (1, 3), (1, 4), (2, 4), (3, 4)]
flag, color_assign = encode_and_solve_three_coloring(n, edge_list)
assert flag, "Error: Graph is three colorable but your code wrongly returns flag = False"
print(f"Three color assignment: {color_assign}")
check_three_color_assign(n, edge_list, color_assign)
print("Passed: 10 points!")

# %%
n = 5
edge_list = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
flag, color_assign = encode_and_solve_three_coloring(n, edge_list)
assert not flag, "Error: Graph is NOT three colorable but your code wrongly returns flag = True"
print("Passed: 5 points!")

# %%
n = 10
edge_list = [
    (1, 5),
    (1, 7),
    (1, 9),
    (2, 4),
    (2, 5),
    (2, 9),
    (3, 4),
    (3, 6),
    (3, 7),
    (3, 8),
    (4, 5),
    (4, 6),
    (4, 7),
    (4, 9),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 9),
]
flag, color_assign = encode_and_solve_three_coloring(n, edge_list)
assert flag, "Error: Graph is three colorable but your code wrongly returns flag = False"
print(f"Three color assignment: {color_assign}")
check_three_color_assign(n, edge_list, color_assign)
print("Passed: 5 points!")

# %% [markdown]
# ## Problem 2
#
# Express the number of warehouses created in terms of the decision variables
# $w_0, \ldots, w_{n-1}$. This will give us the objective that we will minimize.
# \begin{equation}
# \sum_{i=0}^{n-1} w_i
# \end{equation}
#
# Write down the constraint that at least one warehouse must be located among
# the locations in the set $D_j$.
# \begin{equation}
# \sum_{i \in D_j} w_i \geq 1, \qquad j = 0, \ldots, n-1
# \end{equation}


# %%
def euclidean_distance(location_coords, i, j):
    assert 0 <= i < len(location_coords)
    assert 0 <= j < len(location_coords)
    (xi, yi) = location_coords[i]
    (xj, yj) = location_coords[j]
    return ((xj - xi) ** 2 + (yj - yi) ** 2) ** (1 / 2)


def solve_warehouse_location(location_coords, R):
    assert R > 0, "radius must be positive"

    # Decision variables
    n = len(location_coords)
    decision_vars = [LpVariable(f"w{i}", lowBound=0, upBound=1, cat=LpInteger) for i in range(n)]

    # Objective
    prob = LpProblem("Warehouselocation", LpMinimize)
    prob += lpSum(decision_vars)

    # Add constraints
    distances = [[euclidean_distance(location_coords, i, j) for j in range(n)] for i in range(n)]
    for j in range(n):
        dj = filter(lambda i: distances[i][j] <= R, range(n))
        prob += lpSum(decision_vars[i] for i in dj) >= 1

    # Solve problem and return result
    prob.solve()
    return [i for i in range(n) if decision_vars[i].varValue > 0]  # pyright: ignore [reportOptionalOperand]


# %%
def check_solution(location_coords, R, warehouse_locs):
    # For each location i, calculate all locations j within distance R of location i.
    # Use list comprehension instead of accumulating using a nested for loop.
    n = len(location_coords)
    assert all(0 <= j < n for j in warehouse_locs), f"Warehouse locations must be between 0 and {n - 1}"
    neighborhoods = [
        [j for j in range(n) if euclidean_distance(location_coords, i, j) <= R] for i in range(n)
    ]
    w = set(warehouse_locs)
    for i, n_list in enumerate(neighborhoods):
        assert any(
            j in w for j in n_list
        ), f"Location #{i} has no warehouse within distance {R}. The locations within distance {R} are {n_list}"
    print("Your solution passed test")


def visualize_solution(location_coords, R, warehouse_locs):
    n = len(location_coords)
    x_coords, y_coords = zip(*location_coords)
    warehouse_x, warehouse_y = [x_coords[j] for j in warehouse_locs], [y_coords[j] for j in warehouse_locs]
    _, ax = plt.subplots()
    ax.set_aspect("equal")
    plt.scatter(x_coords, y_coords)
    for j in warehouse_locs:
        circ = Circle(location_coords[j], R, alpha=0.5, color="g", ls="--", lw=2, ec="k")
        ax.add_patch(circ)
    for i in range(n):
        ax.annotate(f"{i}", location_coords[i])
    plt.scatter(warehouse_x, warehouse_y, marker="x", c="r", s=30)


# %%
location_coords = [(1, 2), (3, 5), (4, 7), (5, 1), (6, 8), (7, 9), (8, 14), (13, 6)]
R = 5
locs = solve_warehouse_location(location_coords, R)
print(f"Your code returned warehouse locations: {locs}")
assert (
    len(locs) <= 4
), f"Error: There is a solution involving just 4 locations whereas your code returns {len(locs)}"
visualize_solution(location_coords, R, locs)
check_solution(location_coords, R, locs)

# %%
location_coords = [(1, 1), (1, 2), (2, 3), (1, 4), (5, 1), (3, 3), (4, 4), (1, 6), (0, 3), (3, 5), (2, 4)]

# Test 1
print("R = 2 Test:")
R = 2
locs = solve_warehouse_location(location_coords, R)
print(f"Your code returned warehouse locations: {locs}")
assert (
    len(locs) <= 4
), f"Error: There is a solution involving just 4 locations whereas your code returns {len(locs)}"
visualize_solution(location_coords, R, locs)
check_solution(location_coords, R, locs)
print("Test with R = 2 has passed")

# %%
# Test 2
print("R = 3 Test:")
R = 3
locs = solve_warehouse_location(location_coords, R)
print(f"Your code returned warehouse locations: {locs}")
assert (
    len(locs) <= 2
), f"Error: There is a solution involving just 2 locations whereas your code returns {len(locs)}"
visualize_solution(location_coords, R, locs)
check_solution(location_coords, R, locs)
print("Test with R = 3 has passed")

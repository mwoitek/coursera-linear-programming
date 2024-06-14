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

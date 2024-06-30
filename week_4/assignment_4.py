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
# # Week 4 Programming Assignment
# ## Imports

# %%
from random import randint, uniform

from pulp import *  # pyright: ignore [reportWildcardImportFromLibrary]

# %% [markdown]
# ## Problem 1


# %%
def k_tsp_mtz_encoding(n, k, cost_matrix):
    # Check inputs are OK
    assert 1 <= k < n
    assert len(cost_matrix) == n, f"Cost matrix is not {n}x{n}"
    assert all(len(row) == n for row in cost_matrix), f"Cost matrix is not {n}x{n}"

    # Create decision variables
    # Binary variables
    binary_vars = [
        [LpVariable(f"x_{{{i},{j}}}", cat=LpBinary) if i != j else None for j in range(n)] for i in range(n)
    ]
    # Time stamp variables
    time_stamps = [LpVariable(f"t_{i}", lowBound=0, upBound=n) for i in range(1, n)]

    # Create the problem
    prob = LpProblem("kTSP", LpMinimize)

    # Add the objective function
    prob += lpSum(
        lpSum(cost_matrix[i][j] * binary_vars[i][j] for j in filter(lambda x: x != i, range(n)))
        for i in range(n)
    )

    # Add the degree constraints
    # Vertex 0
    num_enter = lpSum(binary_vars[j][0] for j in range(1, n))
    num_leave = lpSum(binary_vars[0][j] for j in range(1, n))
    prob += num_enter == k
    prob += num_leave == k

    # All the other vertices
    for i in range(1, n):
        num_enter = lpSum(binary_vars[j][i] for j in filter(lambda x: x != i, range(n)))
        num_leave = lpSum(binary_vars[i][j] for j in filter(lambda x: x != i, range(n)))
        prob += num_enter == 1
        prob += num_leave == 1

    # Add time stamp constraints
    for i in range(1, n):
        for j in filter(lambda x: x != i, range(1, n)):
            x = binary_vars[i][j]
            assert x is not None
            prob += time_stamps[j - 1] >= time_stamps[i - 1] + x - (1 - x) * (n + 1)

    # Solve the problem
    prob.solve(PULP_CBC_CMD(msg=False))
    status = LpStatus[prob.status]
    assert status == "Optimal", f"Unexpected non-optimal status: {status}"

    # Extract tours
    tours = []

    # Get all second vertices
    second = [j for j, x in enumerate(binary_vars[0]) if x is not None and x.varValue > 0]  # pyright: ignore [reportOptionalOperand]
    assert len(second) == k, "Could not find second vertex for each salesman"

    for v in second:
        tour = [0, v]

        while True:
            nxt = [j for j, x in enumerate(binary_vars[tour[-1]]) if x is not None and x.varValue > 0]  # pyright: ignore [reportOptionalOperand]
            assert len(nxt) == 1
            if nxt[0] == 0:
                break
            tour.append(nxt[0])

        tours.append(tour)

    return tours


# %%
n = 5
k = 2
cost_matrix = [
    [None, 3, 4, 3, 5],
    [1, None, 2, 4, 1],
    [2, 1, None, 5, 4],
    [1, 1, 5, None, 4],
    [2, 1, 3, 5, None],
]
all_tours = k_tsp_mtz_encoding(n, k, cost_matrix)
print(f"Your code returned tours: {all_tours}")
assert len(all_tours) == k, f"k={k} must yield {k} tours -- your code returns {len(all_tours)} tours instead"

tour_cost = 0
for tour in all_tours:
    assert tour[0] == 0, "Each salesperson tour must start from vertex 0"
    i = 0
    for j in tour[1:]:
        tour_cost += cost_matrix[i][j]
        i = j
    tour_cost += cost_matrix[i][0]
print(f"Tour cost obtained by your code: {tour_cost}")
assert abs(tour_cost - 12) <= 0.001, f"Expected tour cost is 12, your code returned {tour_cost}"

for i in range(1, n):
    is_in_tour = [i in tour for tour in all_tours]
    assert sum(is_in_tour) == 1, f"Vertex {i} is in {sum(is_in_tour)} tours -- this is incorrect"

print("Test passed: 3 points")

# %%
n = 5
k = 3
cost_matrix = [
    [None, 3, 4, 3, 5],
    [1, None, 2, 4, 1],
    [2, 1, None, 5, 4],
    [1, 1, 5, None, 4],
    [2, 1, 3, 5, None],
]
all_tours = k_tsp_mtz_encoding(n, k, cost_matrix)
print(f"Your code returned tours: {all_tours}")
assert len(all_tours) == k, f"k={k} must yield {k} tours -- your code returns {len(all_tours)} tours instead"

tour_cost = 0
for tour in all_tours:
    assert tour[0] == 0, "Each salesperson tour must start from vertex 0"
    i = 0
    for j in tour[1:]:
        tour_cost += cost_matrix[i][j]
        i = j
    tour_cost += cost_matrix[i][0]
print(f"Tour cost obtained by your code: {tour_cost}")
assert abs(tour_cost - 16) <= 0.001, f"Expected tour cost is 16, your code returned {tour_cost}"

for i in range(1, n):
    is_in_tour = [i in tour for tour in all_tours]
    assert sum(is_in_tour) == 1, f"Vertex {i} is in {sum(is_in_tour)} tours -- this is incorrect"

print("Test passed: 2 points")

# %%
n = 8
k = 2
cost_matrix = [
    [None, 1, 1, 1, 1, 1, 1, 1],
    [0, None, 1, 2, 1, 1, 1, 1],
    [1, 0, None, 1, 2, 2, 2, 1],
    [1, 2, 2, None, 0, 1, 2, 1],
    [1, 1, 1, 1, None, 1, 1, 1],
    [0, 1, 2, 1, 1, None, 1, 1],
    [1, 0, 1, 2, 2, 2, None, 1],
    [1, 2, 2, 0, 1, 2, 1, None],
]
all_tours = k_tsp_mtz_encoding(n, k, cost_matrix)
print(f"Your code returned tours: {all_tours}")
assert len(all_tours) == k, f"k={k} must yield {k} tours -- your code returns {len(all_tours)} tours instead"

tour_cost = 0
for tour in all_tours:
    assert tour[0] == 0, "Each salesperson tour must start from vertex 0"
    i = 0
    for j in tour[1:]:
        tour_cost += cost_matrix[i][j]
        i = j
    tour_cost += cost_matrix[i][0]
print(f"Tour cost obtained by your code: {tour_cost}")
assert abs(tour_cost - 4) <= 0.001, f"Expected tour cost is 4, your code returned {tour_cost}"

for i in range(1, n):
    is_in_tour = [i in tour for tour in all_tours]
    assert sum(is_in_tour) == 1, f"Vertex {i} is in {sum(is_in_tour)} tours -- this is incorrect"

print("Test passed: 3 points")

# %%
n = 8
k = 4
cost_matrix = [
    [None, 1, 1, 1, 1, 1, 1, 1],
    [0, None, 1, 2, 1, 1, 1, 1],
    [1, 0, None, 1, 2, 2, 2, 1],
    [1, 2, 2, None, 0, 1, 2, 1],
    [1, 1, 1, 1, None, 1, 1, 1],
    [0, 1, 2, 1, 1, None, 1, 1],
    [1, 0, 1, 2, 2, 2, None, 1],
    [1, 2, 2, 0, 1, 2, 1, None],
]
all_tours = k_tsp_mtz_encoding(n, k, cost_matrix)
print(f"Your code returned tours: {all_tours}")
assert len(all_tours) == k, f"k={k} must yield {k} tours -- your code returns {len(all_tours)} tours instead"

tour_cost = 0
for tour in all_tours:
    assert tour[0] == 0, "Each salesperson tour must start from vertex 0"
    i = 0
    for j in tour[1:]:
        tour_cost += cost_matrix[i][j]
        i = j
    tour_cost += cost_matrix[i][0]
print(f"Tour cost obtained by your code: {tour_cost}")
assert abs(tour_cost - 6) <= 0.001, f"Expected tour cost is 6, your code returned {tour_cost}"

for i in range(1, n):
    is_in_tour = [i in tour for tour in all_tours]
    assert sum(is_in_tour) == 1, f"Vertex {i} is in {sum(is_in_tour)} tours -- this is incorrect"

print("Test passed: 2 points")


# %%
def create_cost(n):
    return [[uniform(0, 5) if i != j else None for j in range(n)] for i in range(n)]


for trial in range(5):
    print(f"Trial #{trial}")

    n = randint(5, 11)
    k = randint(2, n // 2)
    print(f"n={n}, k={k}")

    cost_matrix = create_cost(n)
    print("cost_matrix =")
    print(cost_matrix)

    all_tours = k_tsp_mtz_encoding(n, k, cost_matrix)
    print(f"Your code returned tours: {all_tours}")
    assert (
        len(all_tours) == k
    ), f"k={k} must yield {k} tours -- your code returns {len(all_tours)} tours instead"

    tour_cost = 0
    for tour in all_tours:
        assert tour[0] == 0, "Each salesperson tour must start from vertex 0"
        i = 0
        for j in tour[1:]:
            tour_cost += cost_matrix[i][j]
            i = j
        tour_cost += cost_matrix[i][0]  # pyright: ignore
    print(f"Tour cost obtained by your code: {tour_cost}")

    for i in range(1, n):
        is_in_tour = [i in tour for tour in all_tours]
        assert sum(is_in_tour) == 1, f"Vertex {i} is in {sum(is_in_tour)} tours -- this is incorrect"

    print("----------")

print("Test passed: 15 points")

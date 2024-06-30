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

import matplotlib.pyplot as plt
import networkx as nx
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

# %% [markdown]
# ## Problem 1B


# %%
def upto_k_tsp_mtz_encoding(n, k, cost_matrix):
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
    num_leave = lpSum(binary_vars[0][j] for j in range(1, n))
    num_enter = lpSum(binary_vars[j][0] for j in range(1, n))
    prob += num_leave <= k  # At most k leave
    prob += num_enter == num_leave  # All that leave must return

    # All the other vertices
    for i in range(1, n):
        num_leave = lpSum(binary_vars[i][j] for j in filter(lambda x: x != i, range(n)))
        num_enter = lpSum(binary_vars[j][i] for j in filter(lambda x: x != i, range(n)))
        prob += num_leave == 1
        prob += num_enter == 1

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
    assert len(second) <= k

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
k = 3
cost_matrix = [
    [None, 3, 4, 3, 5],
    [1, None, 2, 4, 1],
    [2, 1, None, 5, 4],
    [1, 1, 5, None, 4],
    [2, 1, 3, 5, None],
]
all_tours = upto_k_tsp_mtz_encoding(n, k, cost_matrix)
print(f"Your code returned tours: {all_tours}")
assert len(all_tours) <= k, f"<= {k} tours -- your code returns {len(all_tours)} tours instead"

tour_cost = 0
for tour in all_tours:
    assert tour[0] == 0, "Each salesperson tour must start from vertex 0"
    i = 0
    for j in tour[1:]:
        tour_cost += cost_matrix[i][j]
        i = j
    tour_cost += cost_matrix[i][0]

assert (
    len(all_tours) == 1
), f"In this example, just one salesperson is needed to optimally visit all vertices. Your code returns {len(all_tours)}"
print(f"Tour cost obtained by your code: {tour_cost}")
assert abs(tour_cost - 10) <= 0.001, f"Expected tour cost is 10, your code returned {tour_cost}"

for i in range(1, n):
    is_in_tour = [i in tour for tour in all_tours]
    assert sum(is_in_tour) == 1, f"Vertex {i} is in {sum(is_in_tour)} tours -- this is incorrect"

print("Test passed: 3 points")

# %%
n = 8
k = 5
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
all_tours = upto_k_tsp_mtz_encoding(n, k, cost_matrix)
print(f"Your code returned tours: {all_tours}")
assert len(all_tours) <= k, f"<= {k} tours -- your code returns {len(all_tours)} tours instead"

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
for trial in range(20):
    print(f"Trial #{trial}")

    n = randint(5, 11)
    k = randint(2, n // 2)
    print(f"n={n}, k={k}")

    cost_matrix = create_cost(n)
    print("cost_matrix =")
    print(cost_matrix)

    all_tours = upto_k_tsp_mtz_encoding(n, k, cost_matrix)
    print(f"Your code returned tours: {all_tours}")
    assert len(all_tours) <= k, f"<= {k} tours -- your code returns {len(all_tours)} tours instead"

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

print("Test passed: 4 points")

# %% [markdown]
# ## Problem 2

# %%
# Write down the cost matrix as a list of lists
large_number = 10**9
cost_matrix = [
    [None, 1, 50, 51, large_number],
    [1, None, 2, 52, 53],
    [50, 2, None, 3, 54],
    [51, 52, 3, None, 4],
    [large_number, 53, 54, 4, None],
]

# %% [markdown]
# The following code is not part of the assignment solution.

# %%
G = nx.Graph()
for i in range(4):
    for j in range(i + 1, 5):
        G.add_edge(i, j, weight=cost_matrix[i][j])

# %%
_, ax = plt.subplots(layout="tight")
pos = nx.shell_layout(G)
nx.draw_networkx(G, pos, ax=ax)
edge_labels = {(u, v): G.edges[u, v]["weight"] for u, v in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
plt.show()

# %%
T = nx.minimum_spanning_tree(G)

# %%
_, ax = plt.subplots(layout="tight")
pos = nx.shell_layout(T)
nx.draw_networkx(T, pos, ax=ax)
edge_labels = {(u, v): T.edges[u, v]["weight"] for u, v in T.edges()}
nx.draw_networkx_edge_labels(T, pos, edge_labels=edge_labels, ax=ax)
plt.show()

# %% [markdown]
# Back to the assignment code:

# %%
# Check that the cost matrix is symmetric
assert len(cost_matrix) == 5, f"Cost matrix must have 5 rows. Yours has {len(cost_matrix)} rows."
assert all(len(row) == 5 for row in cost_matrix), f"Each row of the cost matrix must have 5 entries."
for i in range(5):
    for j in range(i):
        assert (
            cost_matrix[i][j] == cost_matrix[j][i]
        ), f"Cost matrix fails to be symmetric at entries {(i, j)} and {(j, i)}"
print("Structure of your cost matrix looks OK (3 points).")


# %%
# MST based TSP approximation
# This code implements the simple MST based shortcutting approach that would
# yield factor of 2 approximation for metric TSPs
def minimum_spanning_tree_tsp(n, cost_matrix):
    G = nx.Graph()
    for i in range(n):
        for j in range(i):
            G.add_edge(i, j, weight=cost_matrix[i][j])
    T = nx.minimum_spanning_tree(G)
    print(f"MST for your graph has the edges {T.edges}")
    mst_cost = 0
    mst_dict = {}  # Store MST as a dictionary
    for i, j in T.edges:
        mst_cost += cost_matrix[i][j]
        if i in mst_dict:
            mst_dict[i].append(j)
        else:
            mst_dict[i] = [j]
        if j in mst_dict:
            mst_dict[j].append(i)
        else:
            mst_dict[j] = [i]
    print(f"MST cost: {mst_cost}")
    print(mst_dict)

    # Form a tour with shortcutting
    def traverse_mst(tour_so_far, cur_node):
        assert cur_node in mst_dict
        next_nodes = mst_dict[cur_node]
        for j in next_nodes:
            if j in tour_so_far:
                continue
            tour_so_far.append(j)
            traverse_mst(tour_so_far, j)

    tour = [0]
    traverse_mst(tour, 0)
    i = 0
    tour_cost = 0
    for j in tour[1:]:
        tour_cost += cost_matrix[i][j]
        i = j
    tour_cost += cost_matrix[i][0]

    return tour, tour_cost


# %%
# Optimal TSP tour using MTZ encoding
def mtz_encoding_tsp(n, cost_matrix):
    assert len(cost_matrix) == n, f"Cost matrix is not {n}x{n}"
    assert all(len(row) == n for row in cost_matrix), f"Cost matrix is not {n}x{n}"
    # Create encoding variables
    binary_vars = [
        [LpVariable(f"x_{i}_{j}", cat="Binary") if i != j else None for j in range(n)] for i in range(n)
    ]
    # Add time stamps
    time_stamps = [LpVariable(f"t_{i}", lowBound=0, upBound=n) for i in range(1, n)]
    # Create the problem
    prob = LpProblem("TSP-MTZ", LpMinimize)
    # Add the objective function
    prob += lpSum(
        lpSum(xij * cj if xij is not None else 0 for xij, cj in zip(brow, crow))
        for brow, crow in zip(binary_vars, cost_matrix)
    )
    # Add the degree constraints
    for i in range(n):
        prob += lpSum(xj for xj in binary_vars[i] if xj is not None) == 1
        prob += lpSum(binary_vars[j][i] for j in range(n) if j != i) == 1
    # Add time stamp constraints
    for i in range(1, n):
        for j in range(1, n):
            if i == j:
                continue
            xij = binary_vars[i][j]
            ti = time_stamps[i - 1]
            tj = time_stamps[j - 1]
            prob += tj >= ti + xij - (1 - xij) * (n + 1)  # pyright: ignore
    # Solve the problem
    status = prob.solve(PULP_CBC_CMD(msg=False))
    assert status == constants.LpStatusOptimal, f"Unexpected non-optimal status {status}"
    # Extract the tour
    tour = [0]
    tour_cost = 0
    while len(tour) < n:
        i = tour[-1]
        sols = [j for j, xij in enumerate(binary_vars[i]) if xij is not None and xij.varValue >= 0.999]  # pyright: ignore
        assert len(sols) == 1
        j = sols[0]
        tour_cost += cost_matrix[i][j]
        tour.append(j)
        assert j != 0
    i = tour[-1]
    tour_cost += cost_matrix[i][0]
    return tour, tour_cost


# %%
# Test that exact answer is 10^6 times smaller than approximate answer
# Compute MST based approximation
tour, tour_cost = minimum_spanning_tree_tsp(5, cost_matrix)
print(f"MST approximation yields tour {tour} with cost {tour_cost}")

# Compute exact answer
opt_tour, opt_tour_cost = mtz_encoding_tsp(5, cost_matrix)
print(f"Optimal tour is {opt_tour} with cost {opt_tour_cost}")

# Check that the fraction is 1 million times apart
assert (
    tour_cost / opt_tour_cost >= 1e06
), f"The TSP + shortcutting tour must be at least 10^6 times costlier than optimum. In your case, the ratio is {tour_cost / opt_tour_cost}"

print("Test passed: 7 points")

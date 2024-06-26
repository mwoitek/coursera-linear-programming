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
# # Week 1: Programming Assignment
# ## Imports

# %%
# It's moronic to do this, but I want to avoid problems with the autograder
from pulp import *  # pyright: ignore [reportWildcardImportFromLibrary]

# %% [markdown]
# ## Problem 1: Use PuLP to encode a linear programming problem


# %%
def formulate_lp_problem(m, n, list_c, list_a, list_b):
    # Assert that the data is compatible
    assert m > 0
    assert n > 0
    assert len(list_c) == n
    assert len(list_a) == m
    assert len(list_a) == len(list_b)
    assert all(len(lst) == n for lst in list_a)

    # Create a linear programming model and set it to maximize its objective
    lp_model = LpProblem("LPProblem", LpMaximize)

    # Create all the decision variables and store them in a list
    decision_vars = [LpVariable(f"x{i}") for i in range(n)]

    # Create the objective function
    lp_model += lpSum([c * v for c, v in zip(list_c, decision_vars)])

    # Create all the constraints
    for coeffs, rhs in zip(list_a, list_b):
        lp_model += lpSum([c * v for c, v in zip(coeffs, decision_vars)]) <= rhs

    # Solve the problem and get its status
    lp_model.solve()
    status = LpStatus[lp_model.status]

    # Return the expected tuple
    is_feasible = False
    is_bounded = False
    opt_sol = []

    if status == "Optimal":
        is_feasible = True
        is_bounded = True
        opt_sol = [value(decision_vars[i]) for i in range(n)]
    elif status == "Unbounded":
        is_feasible = True

    return is_feasible, is_bounded, opt_sol


# %%
# Test 1
m = 4
n = 3
list_c = [1, 1, 1]
list_a = [[2, 1, 2], [1, 0, 0], [0, 1, 0], [0, 0, -1]]
list_b = [5, 7, 9, 4]
is_feas, is_bnded, sols = formulate_lp_problem(m, n, list_c, list_a, list_b)
assert is_feas, "The LP should be feasible -- your code returns infeasible"
assert is_bnded, "The LP should be bounded -- your code returns unbounded"
print(sols)
assert sols[0] is not None
assert sols[1] is not None
assert sols[2] is not None
assert abs(sols[0] - 2.0) <= 1e-04, "x0 must be 2.0"
assert abs(sols[1] - 9.0) <= 1e-04, "x1 must be 9.0"
assert abs(sols[2] + 4.0) <= 1e-04, "x2 must be -4.0"
print("Passed: 3 points!")

# %%
# Test 2: Unbounded problem
m = 5
n = 4
list_c = [-1, 2, 1, 1]
list_a = [[1, 0, -1, 2], [2, -1, 0, 1], [1, 1, 1, 1], [1, -1, 1, 1], [0, -1, 0, 1]]
list_b = [3, 4, 5, 2.5, 3]
is_feas, is_bnded, sols = formulate_lp_problem(m, n, list_c, list_a, list_b)
assert is_feas, "The LP should be feasible. But your code returns a status of infeasible."
assert not is_bnded, "The LP should be unbounded but your code returns a status of bounded."
print("Passed: 3 points")

# %%
# Test 3: Infeasible problem
m = 4
n = 3
list_c = [1, 1, 1]
list_a = [[-2, -1, -2], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
list_b = [-8, 1, 1, 1]
is_feas, is_bnded, sols = formulate_lp_problem(m, n, list_c, list_a, list_b)
assert not is_feas, "The LP should be infeasible -- your code returns feasible"
print("Passed: 3 points!")

# %%
# Test 4
m = 16
n = 15
list_c = [1] * n
list_c[6] = list_c[6] + 1
list_a = []
list_b = []
for i in range(n):
    lst = [0] * n
    lst[i] = -1
    list_a.append(lst)
    list_b.append(0)
list_a.append([1] * n)
list_b.append(1)
is_feas, is_bnded, sols = formulate_lp_problem(m, n, list_c, list_a, list_b)
assert is_feas, "Problem is feasible but your code returned infeasible"
assert is_bnded, "Problem is bounded but your code returned unbounded"
print(sols)
assert sols[6] is not None
assert abs(sols[6] - 1.0) <= 1e-03, "Solution does not match expected one"
assert all(abs(sols[i]) <= 1e-03 for i in range(n) if i != 6), "Solution does not match expected one"  # pyright: ignore [reportArgumentType]
print("Passed: 3 points!")

# %% [markdown]
# ## Problem 2: LP formulation for an investment problem
#
# Write down the expression for the objective function in terms of
# $x_1, \ldots, x_6$. Also specify if we are to maximize or minimize it.
# \begin{equation}
# \max \qquad 25 x_1 + 20 x_2 + 3 x_3 + 1.5 x_4 + 3 x_5 + 4.5 x_6
# \end{equation}
#
# Write down the constraint that expresses that the total cost of investment
# must be less than $B = 10,000$.
# \begin{equation}
# 129 x_1 + 286 x_2 + 72.29 x_3 + 38 x_4 + 52 x_5 + 148 x_6 \leq 10,000
# \end{equation}
#
# Write down the constraints that the total investment in each category cannot
# exceed 2/3 of the budget. You should write down three constraints, one for
# each category.
# \begin{eqnarray}
#   387 x_1 + 858 x_2 &\leq& 20,000 \\
#   216.87 x_3 + 114 x_4 &\leq& 20,000 \\
#   156 x_5 + 444 x_6 &\leq& 20,000
# \end{eqnarray}
#
# Write down the constraints that the total investment in each category must
# exceed 1/6 of the budget. You should write down three constraints, one for
# each category.
# \begin{eqnarray}
#   774 x_1 + 1,716 x_2 &\geq& 10,000 \\
#   433.74 x_3 + 228 x_4 &\geq& 10,000 \\
#   312 x_5 + 888 x_6 &\geq& 10,000
# \end{eqnarray}
#
# Write down an expression for the price of the overall portfolio. Also write
# down an expression for the overall earnings of the portfolio.
# \begin{eqnarray}
#   \mathrm{Price} &=& 129 x_1 + 286 x_2 + 72.29 x_3 + 38 x_4 + 52 x_5 + 148 x_6 \\
#   \mathrm{Earnings} &=& 1.9 x_1 + 8.1 x_2 + 1.5 x_3 + 5 x_4 + 2.5 x_5 + 5.2 x_6
# \end{eqnarray}
#
# We wish to enforce the constraint that the overall Price/Earnings ratio of
# the portfolio cannot exceed 15. Write down the constraint as
# $\mathrm{Price} \leq 15 \times \mathrm{Earnings}$.
# \begin{equation}
# 100.5 x_1 + 164.5 x_2 + 49.79 x_3 - 37 x_4 + 14.5 x_5 + 70 x_6 \leq 0
# \end{equation}

# %%
# Create a linear programming model and set it to maximize its objective
lpModel = LpProblem("InvestmentProblem", LpMaximize)

# Create a variable called x1 and set its bounds to be between 0 and infinity
x1 = LpVariable("x1", 0)

# Next create variables x2, ..., x6
x2 = LpVariable("x2", 0)
x3 = LpVariable("x3", 0)
x4 = LpVariable("x4", 0)
x5 = LpVariable("x5", 0)
x6 = LpVariable("x6", 0)

# Set the objective function
lpModel += 25 * x1 + 20 * x2 + 3 * x3 + 1.5 * x4 + 3 * x5 + 4.5 * x6

# Add the constraints
lpModel += 129 * x1 + 286 * x2 + 72.29 * x3 + 38 * x4 + 52 * x5 + 148 * x6 <= 10000
lpModel += 387 * x1 + 858 * x2 <= 20000
lpModel += 216.87 * x3 + 114 * x4 <= 20000
lpModel += 156 * x5 + 444 * x6 <= 20000
lpModel += 774 * x1 + 1716 * x2 >= 10000
lpModel += 433.74 * x3 + 228 * x4 >= 10000
lpModel += 312 * x5 + 888 * x6 >= 10000
lpModel += 100.5 * x1 + 164.5 * x2 + 49.79 * x3 - 37 * x4 + 14.5 * x5 + 70 * x6 <= 0

# Solve the model and print the solutions
lpModel.solve()
for v in lpModel.variables():
    print(v.name, "=", v.varValue)

# Optimized objective function
print("Objective value =", value(lpModel.objective))

# %%
assert abs(value(lpModel.objective) - 1098.59) <= 0.1, "Test failed"

# %% [markdown]
# ## Problem 3: Optimal Transport
#
# Write down the objective function in terms of $x_{i,j}$ and $D_{i,j}$ for
# $1 \leq i \leq n$ and $1 \leq j \leq m$. Also indicate if we are going to
# maximize or minimize it.
# \begin{equation}
# \min \qquad \sum_{i=1}^{n} \sum_{j=1}^{m} x_{i,j} D_{i,j}
# \end{equation}
#
# Next, for each source location $i$, the total amount of material transported
# from $i$ to various destination locations must sum up to $w_i$: the total
# weight of material at location $i$. Write down a constraint to enforce this.
# \begin{equation}
# \sum_{j=1}^{m} x_{i,j} = w_i, \qquad 1 \leq i \leq n
# \end{equation}
#
# Next, for each destination location $j$, the total amount of material
# transported to $j$ from various source locations must sum up to $w_{j}^{\prime}$:
# the total weight of material at destination location $j$. Write down a
# constraint to enforce this.
# \begin{equation}
# \sum_{i=1}^{n} x_{i,j} = w_{j}^{\prime}, \qquad 1 \leq j \leq m
# \end{equation}


# %%
def calculate_distance(a, b):
    (xa, ya) = a
    (xb, yb) = b
    return ((xa - xb) ** 2 + (ya - yb) ** 2) ** (1 / 2)


def get_objective(var_values, source_coords, dest_coords):
    n = len(source_coords)
    m = len(dest_coords)
    return sum(
        var_values[i][j] * calculate_distance(source_coords[i], dest_coords[j])
        for i in range(n)
        for j in range(m)
    )


def calculate_optimal_transport_plan(source_coords, source_weights, dest_coords, dest_weights):
    n = len(source_coords)
    assert n == len(source_weights)
    m = len(dest_coords)
    assert m == len(dest_weights)
    assert sum(source_weights) == sum(dest_weights)

    # Create the LP model
    lp_model = LpProblem("OptimalTransport", LpMinimize)

    # Create a list of decision variables x_{i,j}
    decision_vars = [[LpVariable(f"x_{{{i},{j}}}", lowBound=0) for j in range(m)] for i in range(n)]

    # Add the objective function
    lp_model += lpSum(
        decision_vars[i][j] * calculate_distance(source_coords[i], dest_coords[j])
        for i in range(n)
        for j in range(m)
    )

    # Add the constraints
    for i in range(n):
        lp_model += lpSum(decision_vars[i][j] for j in range(m)) == source_weights[i]
    for j in range(m):
        lp_model += lpSum(decision_vars[i][j] for i in range(n)) == dest_weights[j]

    # Solve and return the solution
    lp_model.solve()
    return [[value(decision_vars[i][j]) for j in range(m)] for i in range(n)]


# %%
source_coords = [(1, 5), (4, 1), (5, 5)]
source_weights = [9, 4, 5]
dest_coords = [(2, 2), (6, 6)]
dest_weights = [9, 9]
n = 3
m = 2
var_values = calculate_optimal_transport_plan(source_coords, source_weights, dest_coords, dest_weights)
obj_val = get_objective(var_values, source_coords, dest_coords)
print(f"Objective value: {obj_val}")

# Check the solution
for i in range(n):
    assert sum(var_values[i][j] for j in range(m)) == source_weights[i]  # pyright: ignore
for j in range(m):
    assert sum(var_values[i][j] for i in range(n)) == dest_weights[j]  # pyright: ignore

assert abs(obj_val - 52.22) <= 1e-01
print("Test Passed: 10 points")

# %%
source_coords = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]
source_weights = [10, 10, 10, 10, 10, 10]
dest_coords = [(6, 1), (5, 2), (4, 3), (3, 2), (2, 1)]
dest_weights = [12, 12, 12, 12, 12]
n = 6
m = 5
var_values = calculate_optimal_transport_plan(source_coords, source_weights, dest_coords, dest_weights)
obj_val = get_objective(var_values, source_coords, dest_coords)
print(f"Objective value: {obj_val}")

# Check the solution
for i in range(n):
    assert sum(var_values[i][j] for j in range(m)) == source_weights[i]  # pyright: ignore
for j in range(m):
    assert sum(var_values[i][j] for i in range(n)) == dest_weights[j]  # pyright: ignore

assert abs(obj_val - 127.19) <= 1e-1
print("Test Passed: 8 points")

# %%
source_coords = [(i, 1) for i in range(20)]
source_weights = [10] * 20
dest_coords = [(6, i + 5) for i in range(8)] + [(14, i + 5) for i in range(8)]
dest_weights = [12.5] * 16
n = 20
m = 16
var_values = calculate_optimal_transport_plan(source_coords, source_weights, dest_coords, dest_weights)
obj_val = get_objective(var_values, source_coords, dest_coords)
print(f"Objective value: {obj_val}")

# Check the solution
for i in range(n):
    assert sum(var_values[i][j] for j in range(m)) == source_weights[i]  # pyright: ignore
for j in range(m):
    assert sum(var_values[i][j] for i in range(n)) == dest_weights[j]  # pyright: ignore

assert abs(obj_val - 1598.11) <= 1e-1
print("Test Passed: 5 points")

# %% [markdown]
# ## Problem 3B: Optimal Pricing Model


# %%
def compute_optimal_prices(source_coords, source_weights, dest_coords, dest_weights):
    n = len(source_coords)
    assert n == len(source_weights)
    m = len(dest_coords)
    assert m == len(dest_weights)
    assert sum(source_weights) == sum(dest_weights)

    lp_model = LpProblem("Transportation", LpMaximize)

    # Decision variables
    source_vars = [LpVariable(f"SourcePrice_{i}", lowBound=0) for i in range(n)]
    dest_vars = [LpVariable(f"DestinationPrice_{j}", lowBound=0) for j in range(m)]

    # Objective function
    source_price = lpSum(source_vars[i] * source_weights[i] for i in range(n))
    dest_price = lpSum(dest_vars[j] * dest_weights[j] for j in range(m))
    lp_model += dest_price - source_price

    # Constraints
    for i in range(n):
        for j in range(m):
            lp_model += dest_vars[j] - source_vars[i] <= calculate_distance(source_coords[i], dest_coords[j])

    lp_model.solve()
    return [v.varValue for v in source_vars], [v.varValue for v in dest_vars]


# %%
source_coords = [(1, 5), (4, 1), (5, 5)]
source_weights = [9, 4, 5]
dest_coords = [(2, 2), (6, 6)]
dest_weights = [9, 9]
n = 3
m = 2
source_prices, dest_prices = compute_optimal_prices(source_coords, source_weights, dest_coords, dest_weights)
profit = sum(p * w for p, w in zip(dest_prices, dest_weights)) - sum(  # pyright: ignore
    p * w  # pyright: ignore
    for p, w in zip(source_prices, source_weights)
)
assert abs(profit - 52.22) <= 1e-01, f"Error: Expected profit is 52.22 obtained: {profit}"
print("Test Passed: 7 points")

# %%
source_coords = [(i, 1) for i in range(20)]
source_weights = [10] * 20
dest_coords = [(6, i + 5) for i in range(8)] + [(14, i + 5) for i in range(8)]
dest_weights = [12.5] * 16
n = 20
m = 16
source_prices, dest_prices = compute_optimal_prices(source_coords, source_weights, dest_coords, dest_weights)
profit = sum(p * w for p, w in zip(dest_prices, dest_weights)) - sum(  # pyright: ignore
    p * w  # pyright: ignore
    for p, w in zip(source_prices, source_weights)
)
assert abs(profit - 1598.11) <= 1e-1, f"Error: Expected profit is 1598.11 obtained: {profit}"
print("Test Passed: 8 points")

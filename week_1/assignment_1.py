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
import re

import pulp as pl

# %% [markdown]
# ## Problem 1: Use PuLP to encode a linear programming problem


# %%
# Define a key function to extract the numeric part of each variable name
def extract_number(lp_var):
    match_obj = re.search(r"\d+", lp_var.name)
    assert match_obj is not None
    return int(match_obj.group())


def formulate_lp_problem(m, n, list_c, list_a, list_b):
    # Assert that the data is compatible
    assert m > 0
    assert n > 0
    assert len(list_c) == n
    assert len(list_a) == m
    assert len(list_a) == len(list_b)
    assert all(len(lst) == n for lst in list_a)

    # Create a linear programming model and set it to maximize its objective
    lp_model = pl.LpProblem("LPProblem", pl.LpMaximize)

    # Create all the decision variables and store them in a list
    decision_vars = [pl.LpVariable(f"x{i}") for i in range(n)]

    # Create the objective function
    lp_model += pl.lpSum([c * v for c, v in zip(list_c, decision_vars)])

    # Create all the constraints
    for coeffs, rhs in zip(list_a, list_b):
        lhs = pl.lpSum([c * v for c, v in zip(coeffs, decision_vars)])
        lp_model += lhs <= rhs

    # Solve the problem and get its status
    lp_model.solve()
    status = pl.LpStatus[lp_model.status]

    # Return the expected tuple
    is_feasible = False
    is_bounded = False
    opt_sol = []

    if status == "Optimal":
        is_feasible = True
        is_bounded = True
        # Get the model variables in the right order
        lp_vars = sorted(lp_model.variables(), key=extract_number)
        opt_sol = [lp_var.varValue for lp_var in lp_vars]
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
assert abs(sols[6] - 1.0) <= 1e-03, "Solution does not match expected one"
assert all([abs(sols[i]) <= 1e-03 for i in range(n) if i != 6]), "Solution does not match expected one"
print("Passed: 3 points!")

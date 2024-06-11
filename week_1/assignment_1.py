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

# It's moronic to do this, but I want to avoid problems with the autograder
from pulp import *  # pyright: ignore [reportWildcardImportFromLibrary]

# %% [markdown]
# ## Problem 1: Use PuLP to encode a linear programming problem


# %%
# Define a key function to extract the numeric part of each variable name
def extract_number(lp_var):
    return int(re.search(r"\d+", lp_var.name).group())  # pyright: ignore [reportOptionalMemberAccess]


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
        lhs = lpSum([c * v for c, v in zip(coeffs, decision_vars)])
        lp_model += lhs <= rhs

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

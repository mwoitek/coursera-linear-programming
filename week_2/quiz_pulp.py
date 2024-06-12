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
# # Quiz: Formulating/Solving ILPs
# ## Imports

# %%
import pulp as pl

# %% [markdown]
# ## Problem 1

# %%
model_1 = pl.LpProblem("Problem1", pl.LpMaximize)
vars_1 = [pl.LpVariable(f"x{i}", lowBound=-15, upBound=15, cat=pl.LpInteger) for i in range(5)]

# Objective function
model_1 += 2 * vars_1[0] - 3 * vars_1[1] + vars_1[2]

# Constraints
model_1 += vars_1[0] - vars_1[1] + vars_1[2] <= 5
model_1 += vars_1[0] - vars_1[1] + 4 * vars_1[2] <= 7
model_1 += vars_1[0] + 2 * vars_1[1] - vars_1[2] + vars_1[3] <= 14
model_1 += vars_1[2] - vars_1[3] + vars_1[4] <= 7

model_1.solve()

for var in vars_1:
    print(var.name, "=", var.varValue)

print(f"Optimal value = {model_1.objective.value():.2f}")  # Optimal value = 40.00

# %% [markdown]
# ## Problem 2

# %%
model_2 = pl.LpProblem("Problem2", pl.LpMaximize)
vars_2 = [pl.LpVariable(f"x{i}", lowBound=-15, upBound=15) for i in range(5)]

# Objective function
model_2 += 2 * vars_2[0] - 3 * vars_2[1] + vars_2[2]

# Constraints
model_2 += vars_2[0] - vars_2[1] + vars_2[2] <= 5
model_2 += vars_2[0] - vars_2[1] + 4 * vars_2[2] <= 7
model_2 += vars_2[0] + 2 * vars_2[1] - vars_2[2] + vars_2[3] <= 14
model_2 += vars_2[2] - vars_2[3] + vars_2[4] <= 7

model_2.solve()

for var in vars_2:
    print(var.name, "=", var.varValue)

print(f"Optimal value = {model_2.objective.value():.2f}")  # Optimal value = 40.00

# %% [markdown]
# ## Problem 3
# - They have the same solution.
# - The integrality gap is 1.
# - The LP relaxation yields an integral solution for all the decision
#   variables and we know from just that information that the ILP will also
#   have the same optimal solution.
#
# ## Problem 4

# %%
model_3 = pl.LpProblem("Problem4", pl.LpMinimize)
vars_3 = [pl.LpVariable(f"x{i}", lowBound=-1, upBound=1, cat=pl.LpInteger) for i in range(3)]

# Objective function
model_3 += 2 * vars_3[0] - 3 * vars_3[1] + vars_3[2]

# Constraint
model_3 += vars_3[0] - vars_3[1] >= 0.5
model_3 += vars_3[0] - vars_3[1] <= 0.75
model_3 += vars_3[1] - vars_3[2] <= 1.25
model_3 += vars_3[1] - vars_3[2] >= 0.95

model_3.solve()
print(f"Status = {pl.LpStatus[model_3.status]}")  # Status = Infeasible

# %% [markdown]
# ## Problem 5

# %%
model_4 = pl.LpProblem("Problem5", pl.LpMinimize)
vars_4 = [pl.LpVariable(f"x{i}", lowBound=-1, upBound=1) for i in range(3)]

# Objective function
model_4 += 2 * vars_4[0] - 3 * vars_4[1] + vars_4[2]

# Constraint
model_4 += vars_4[0] - vars_4[1] >= 0.5
model_4 += vars_4[0] - vars_4[1] <= 0.75
model_4 += vars_4[1] - vars_4[2] <= 1.25
model_4 += vars_4[1] - vars_4[2] >= 0.95

model_4.solve()
print(f"Status = {pl.LpStatus[model_4.status]}")  # Status = Optimal

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
# # Quiz: Vertex Cover ILP, LP Relaxation and Integrality Gap
# ## Imports

# %%
import pulp as pl

# %% [markdown]
# ## Problem 1
# - Using the rounding procedure on fractional solutions of the LP relaxation
#   would yield a vertex cover of size between 442 and 884.
# - The optimal vertex cover size lies in the range [442, 884].
#
# ## Problem 2

# %%
edges = [
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 3),
    (2, 6),
    (3, 4),
    (3, 5),
    (3, 6),
    (4, 5),
    (4, 7),
    (5, 6),
    (5, 7),
    (6, 7),
]

# %%
prob = pl.LpProblem("VertexCoverILP", pl.LpMinimize)

# Decision variables
x = [pl.LpVariable(f"x{i}", lowBound=0, upBound=1, cat=pl.LpInteger) for i in range(7)]

# Objective
prob += pl.lpSum(x)

# Constraints
for u, v in ((u - 1, v - 1) for u, v in edges):
    prob += x[u] + x[v] >= 1

prob.solve()
assert pl.LpStatus[prob.status] == "Optimal"

# %%
vertex_cover = [i + 1 for i in range(7) if x[i].varValue > 0]  # pyright: ignore
vertex_cover

# %%
len(vertex_cover)

# %% [markdown]
# ## Problem 3

# %%
prob = pl.LpProblem("VertexCoverLPRelaxation", pl.LpMinimize)

# Decision variables
z = [pl.LpVariable(f"z{i}", lowBound=0, upBound=1) for i in range(7)]

# Objective
prob += pl.lpSum(z)

# Constraints
for u, v in ((u - 1, v - 1) for u, v in edges):
    prob += z[u] + z[v] >= 1

prob.solve()
assert pl.LpStatus[prob.status] == "Optimal"

# %%
pl.value(prob.objective)

# %% [markdown]
# ## Problem 4

# %%
vertex_cover = [i + 1 for i in range(7) if z[i].varValue >= 0.5]  # pyright: ignore
vertex_cover

# %%
len(vertex_cover)

# %% [markdown]
# **All the vertices of the graph are part of this cover.**

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
# # Quiz: Vertex Cover
# ## Problem 1


# %%
class Graph:
    def __init__(self, edges):
        self.num_edges = len(edges)
        self.adj_list = dict()
        self.degrees = dict()

        for edge in edges:
            self.add_edge(edge)

    def add_edge(self, edge):
        u, v = edge
        if v < u:
            u, v = v, u

        if u in self.adj_list:
            self.adj_list[u].add(v)
        else:
            self.adj_list[u] = {v}

        if u in self.degrees:
            self.degrees[u] += 1
        else:
            self.degrees[u] = 1

        if v in self.degrees:
            self.degrees[v] += 1
        else:
            self.degrees[v] = 1

    def max_degree_vertex(self):
        return max(self.degrees.items(), key=lambda p: (p[1], -p[0]))[0]

    def remove_vertex(self, vertex):
        self.num_edges -= self.degrees[vertex]
        del self.degrees[vertex]

        if vertex in self.adj_list:
            for v in self.adj_list[vertex]:
                self.degrees[v] -= 1
            del self.adj_list[vertex]

        v = vertex - 1
        while v >= 1:
            if self.has_edge((v, vertex)):
                self.adj_list[v].remove(vertex)
                self.degrees[v] -= 1
            v -= 1

    def has_edge(self, edge):
        u, v = edge
        if v < u:
            u, v = v, u
        return u in self.adj_list and v in self.adj_list[u]

    def has_edges(self):
        return self.num_edges > 0

    def vertex_cover_max_degree(self):
        vertex_cover = []

        while self.has_edges():
            vertex = self.max_degree_vertex()
            vertex_cover.append(vertex)
            self.remove_vertex(vertex)

        return vertex_cover

    def vertex_cover_max_matching(self, edges_to_process):
        vertex_cover = []
        i = 0

        while self.has_edges():
            edge = edges_to_process[i]
            if not self.has_edge(edge):
                i += 1
                continue

            u, v = edge
            vertex_cover.append(u)
            vertex_cover.append(v)

            self.remove_vertex(u)
            self.remove_vertex(v)
            i += 1

        return vertex_cover


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
g1 = Graph(edges)
g1.vertex_cover_max_degree()

# %% [markdown]
# - The greedy algorithm produces a vertex cover of size 5 that involves nodes
#   1, 3, 4, 5, 6.
# - Node 6 is the third node added to the cover.
# - Node 3 is the very first node added to the cover.
# - Node 4 is the second node added to the cover.
#
# ## Problem 2

# %%
g2 = Graph(edges)
vertex_cover = g2.vertex_cover_max_matching(edges)
len(vertex_cover)

# %%
vertex_cover

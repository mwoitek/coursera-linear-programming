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
# # Quiz: Job Scheduling
# ## Imports

# %%
import heapq
from dataclasses import dataclass
from operator import itemgetter
from pprint import pprint

# %% [markdown]
# ## Jobs


# %%
@dataclass
class Job:
    id_: int
    time: int


# %%
jobs: list[Job] = [
    Job(id_=1, time=5),
    Job(id_=2, time=6),
    Job(id_=3, time=2),
    Job(id_=4, time=5),
    Job(id_=5, time=2),
    Job(id_=6, time=1),
    Job(id_=7, time=3),
    Job(id_=8, time=4),
]

# %% [markdown]
# ## Problem 1


# %%
def greedy_scheduler(jobs: list[Job], num_procs: int) -> dict[int, list[Job]]:
    num_jobs = len(jobs)
    assign = [-1] * num_jobs

    heap = [(0, i) for i in range(1, num_procs + 1)]
    heapq.heapify(heap)

    for i, job in enumerate(jobs):
        j, k = heapq.heappop(heap)
        assign[i] = k
        heapq.heappush(heap, (j + job.time, k))

    assign_dict: dict[int, list[Job]] = {}

    for job, proc in zip(jobs, assign):
        if proc in assign_dict:
            assign_dict[proc].append(job)
        else:
            assign_dict[proc] = [job]

    return assign_dict


# %%
def print_assignment(assign: dict[int, list[Job]]) -> None:
    for proc, jobs in assign.items():
        ids = ", ".join(str(job.id_) for job in jobs)
        total_time = sum(job.time for job in jobs)
        print(f"Processor {proc} -> Jobs {ids} -- Total time: {total_time}")


# %%
def compute_makespan(assign: dict[int, list[Job]]) -> tuple[int, int]:
    total_times: list[tuple[int, int]] = []
    for proc, jobs in assign.items():
        total_time = sum(job.time for job in jobs)
        total_times.append((total_time, proc))
    return max(total_times, key=itemgetter(0))


# %%
assign_unsorted = greedy_scheduler(jobs, num_procs=3)
print_assignment(assign_unsorted)
makespan_unsorted, proc = compute_makespan(assign_unsorted)
print(f"Makespan = {makespan_unsorted} -- Processor {proc}")

# %% [markdown]
# ## Problem 2

# %%
assign_2 = {
    1: [jobs[1], jobs[2], jobs[4]],
    2: [jobs[0], jobs[5], jobs[6]],
    3: [jobs[3], jobs[7]],
}
makespan_2, proc = compute_makespan(assign_2)
print(f"Makespan = {makespan_2} -- Processor {proc}")

# %% [markdown]
# ## Problem 3

# %%
jobs = sorted(jobs, key=lambda j: j.time, reverse=True)
pprint(jobs)

# %%
assign_sorted = greedy_scheduler(jobs, num_procs=3)
print_assignment(assign_sorted)
makespan_sorted, proc = compute_makespan(assign_sorted)
print(f"Makespan = {makespan_sorted} -- Processor {proc}")

# %% [markdown]
# ## Problem 4
# - The total time for all jobs is 28. No matter how we distribute it among
#   three processors, at least one processor must have finish time more than
#   $\frac{28}{3}$ or $9.3333\ldots$ Therefore, the optimal answer cannot be
#   smaller than 10.
# - By the pigeon hole principle, if we consider the first four job IDs 1, 2,
#   3, 4; at least two jobs must share the same processor in any assignment.
#   This yields a lower bound of 7 on the optimal solution.
# - The following assignment: [same assignment from Problem 2] has a makespan
#   of 10. And therefore the optimal solution must have value $\leq 10$.

import numpy as np
import random

num_nodes = 4
dist = np.asarray([[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]]).astype('float64')

def traverse_graph(source_node):
    cycle = [source_node]

    visited = np.array([1 for _ in range(num_nodes)])
    visited[source_node] = 0
    cur = source_node

    while len(cycle) < num_nodes:
        nearest_node = [-1, 1000000000]
        for node in range(num_nodes):
            if visited[node] != 0:
                if (dist[cur][node] < nearest_node[1]):
                    nearest_node = [node, dist[cur][node]]

        cur = nearest_node[0]
        cycle.append(cur)
        visited[cur] = 0

    total_length = cycle_length(cycle)
    return total_length, cycle

def cycle_length(cycle):
    length = 0
    for i in range(num_nodes):
        length += dist[cycle[i % num_nodes]][cycle[(i + 1) % num_nodes]]
    return length

def main():
    best_len = 100000000
    best_cycle = []
    for i in range(num_nodes):
        ret = traverse_graph(i)
        if ret[0] < best_len:
            best_len = ret[0]
            best_cycle = ret[1]

    print(best_len)
    print(best_cycle)

main()
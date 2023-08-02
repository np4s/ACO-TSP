import random
import numpy as np
from numpy import ndarray

ALPHA = 0.9
BETA = 0.5
best_len = 1000000000
best_cycle = []
num_nodes = 4
dist = np.asarray([[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]]).astype('float64')
intensity = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]).astype('float64')


def run_aco_batch(batch_size):
    evaporate_rate = random.uniform(0.6, 1)
    # print(evaporate_rate)
    # evaporate
    for i in range(num_nodes):
        for j in range(num_nodes):
            intensity[i][j] *= evaporate_rate

    new_intensity: ndarray = intensity

    # run batch
    global best_len
    global best_cycle
    for ant in range(batch_size):
        ret = traverse_graph(random.randrange(num_nodes))  # the ants start traveling at a random node
        path = ret[0]
        l = ret[1]

        if l < best_len:
            best_len = l
            best_cycle = path

        diff = l - best_len + 0.05
        w = 0.01 / diff
        # the ant leaving pheromone
        for i in range(num_nodes):
            idx1 = path[i % num_nodes]
            idx2 = path[(i + 1) % num_nodes]
            new_intensity[idx1][idx2] += w
            new_intensity[idx2][idx1] += w

    # update the pheromone intensity after normalizing
    for i in range(num_nodes):
        n_sum = 0.0
        for j in range(num_nodes):
            if i == j:
                continue
            n_sum += new_intensity[i][j]
        for j in range(num_nodes):
            intensity[i][j] = 2 * new_intensity[i][j] / n_sum

    return best_len


def traverse_graph(source_node):
    visited = np.asarray([1 for _ in range(num_nodes)])
    visited[source_node] = 0
    cycle = [source_node]
    cur = source_node

    while len(cycle) < num_nodes:
        neighbors = []
        prob_sum = 0.0

        for node in range(num_nodes):
            if visited[node] != 0:  # not visited
                neighbors.append(node)
                prob_sum += get_probability(cur, node)

        r = random.uniform(0.0, prob_sum)
        x = 0.0

        # choosing next node
        for node in neighbors:
            x += get_probability(cur, node)
            if r <= x:
                cycle.append(node)
                cur = node
                visited[cur] = 0
                break

    total_length = cycle_length(cycle)
    return cycle, total_length


def get_probability(idx1, idx2):
    return (intensity[idx1][idx2] ** ALPHA) / (dist[idx1][idx2] ** BETA)


def cycle_length(cycle):
    length = 0
    for i in range(num_nodes):
        length += dist[cycle[i % num_nodes]][cycle[(i + 1) % num_nodes]]
    return length

run_aco_batch(4)
print(best_len)
print(best_cycle)
print('This is a new line')
print('This is 2nd new line')
#modules
import numpy as np
import itertools
import random

#submodules
from matplotlib import pyplot as plt


def D(S, T, node_index, shortest_path_matrix, diameter, **kwargs):
    min_dist = []
    T_gene_indexes = [node_index[t] for t in T]
    S_gene_indexes = [node_index[s] for s in S]
    for t in T_gene_indexes:
        distance = min(shortest_path_matrix[t, S_gene_indexes])
        min_dist.append(distance)
    score = np.mean(min_dist)
    return 1 - score/diameter

def exponential_decay_score(S, T, node_index, shortest_path_matrix, **kwargs):
    """
    Computes the exponential decay score:
        1 - Compute the distance between each node in T and S.
            The distance between a node t and a set of nodes S is defined as the shortest path length
            between between t and any node in S.
        2 - If max_distance is the maximal distance between a node t in T and S, count the number
            of nodes in T at each distance i from S where i in 0, ... , max_distance.
        3 - Perform the weighted sum of the counts.
        4 - Normalize with the maximal score achievable obtained by assuming that
            all node of T are at distance zero.
    """
    distances = {}
    max_distance = -1
    T_gene_indexes = [node_index[t] for t in T]
    S_gene_indexes = [node_index[s] for s in S]
    for t in T_gene_indexes:
        # Distance between t and S
        distance = np.min(shortest_path_matrix[t, S_gene_indexes])
        if max_distance < distance:
            max_distance = distance
        if distances.get(distance) is None:
            distances[distance] = 1
        else:
            distances[distance] += 1
    distance_array = np.zeros(max_distance+1) # plus one to cound also distance zero
    for d in distances.keys():
        distance_array[d] = distances[d]
    weigth_array = np.array([np.exp(-d) for d in range(0, max_distance+1)])
    score = np.sum(np.multiply(distance_array, weigth_array))
    max_score = weigth_array[0]*len(T)
    return score/max_score


def best_score(cs, disease_gene, target_gene, precomputed_scores, score_function, 
               min=True, sample_dis=None, sample_tar=None, **kwargs):
    ncm = cs.to_node_community_map()
    ### Extrat the disease modules
    if ncm.get(disease_gene) is not None:
        # List of tuples (module_number, module)
        disease_modules = [(i, set(cs.communities[i])) for i in ncm[disease_gene]]
    else:
        # If the gene is not part of any module then [(None, {gene})]
        disease_modules = [(None, set([disease_gene]))]

    if sample_dis is not None:
        disease_modules = random.sample(disease_modules, sample_dis)

    ### Extract the target modules
    if 'random_modules' in list(kwargs.keys()):
        random_modules = kwargs['random_modules']
        target_modules = [(mod[0], mod[1]) for mod in random_modules[target_gene]]
        if sample_tar is not None:
            target_modules = random.sample(target_modules, sample_tar)
        candidate_score = []
        for (S_index, S), (T_index, T) in itertools.product(disease_modules, target_modules):
            if precomputed_scores.get((S_index, T_index)) is None:
                score = score_function(S, T, **kwargs)
                if S_index is not None and T_index is not None:
                    precomputed_scores[(S_index, T_index)] = score
                candidate_score.append(score)
            else:
                score = precomputed_scores.get((S_index, T_index))
                candidate_score.append(score)
        return min(candidate_score) if min else max(candidate_score)
    else:
        if ncm.get(target_gene) is not None:
            target_modules = [(i, set(cs.communities[i])) for i in ncm[target_gene]]
        else:
            target_modules = [(None, set([target_gene]))]
        if sample_tar is not None:
            target_modules = random.sample(target_modules, sample_tar)

        candidate_score = []
        for (S_index, S), (T_index, T) in itertools.product(disease_modules, target_modules):
            if precomputed_scores.get((S_index, T_index)) is None:
                ### FUNCTION ###
                score = score_function(S, T, **kwargs)
                ### END FUNCTION ###
                if S_index is not None and T_index is not None:
                    precomputed_scores[(S_index, T_index)] = score
                candidate_score.append(score)
            else:
                score = precomputed_scores.get((S_index, T_index))
                candidate_score.append(score)
        return min(candidate_score) if min else max(candidate_score)


def show_score_matrix(score, x_order=None, y_order=None, row_partitioning=None):
    # Display scores
    fig, ax = plt.subplots()
    fig.set_size_inches(100, 10)
    im = ax.imshow(score)

    # Loop over data dimensions and create text annotations.
    for i in range(len(score)):
        for j in range(len(score[i])):
            text = ax.text(j, i, '{:.2f}'.format(score[i, j]),
                           ha="center", va="center", color="w", fontsize = 'xx-large')

    # Add labels
    if x_order is not None:
        x_labels = [[], []]
        for gene, positions in x_order.items():
            x_labels[0].append(positions)
            x_labels[1].append(gene)
        ax.set_xticks(x_labels[0], x_labels[1])
    if y_order is not None:
        y_labels = [[], []]
        for gene, positions in y_order.items():
            y_labels[0].append(positions)
            y_labels[1].append(gene)
        ax.set_yticks(y_labels[0], y_labels[1])

    # Row partitioning
    if row_partitioning is not None:
        for line_position in row_partitioning:
            ax.hlines(line_position + 0.5, -0.5, score.shape[1]-0.5, color='k', linewidth=5)

    fig.tight_layout()
    plt.show()


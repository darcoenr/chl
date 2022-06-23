import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt
import csv
import numpy as np

from copy import deepcopy
import itertools

def get_biogrid_aliases():
    aliases = {}
    with open('aliases_2353463.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) > 2: # This is due to a bug
                continue
            aliases[row[0]] = row[1]
    return aliases

def relabel_node_with_aliases(G):
    # Get the aliases
    aliases = {}
    with open('aliases_2353463.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) > 2: # This is due to a bug
                    continue
            aliases[row[0]] = row[1]
    nx.relabel_nodes(G, aliases, copy=False)

def print_graph_properties(G):
    print('Number of nodes:', G.number_of_nodes())
    print('Number of edges:', G.number_of_edges())
    print('Density: {:.3f}'.format(nx.density(G)))
    print('Is connected:', nx.is_connected(G))
    print('Number of connected components:', nx.number_connected_components(G))
    connected_components = [len(x) for x in sorted(nx.connected_components(G), key=len, reverse=True)]
    number_isolated = len(list(filter(lambda x: x == 1, connected_components)))
    print('Largest connected component:', connected_components[0], '({:.2f}%)'.format(connected_components[0]/G.number_of_nodes()))
    print('Number of isolated nodes:', number_isolated, '({:.2f}%)'.format(number_isolated/G.number_of_nodes()))
    print('Number of self loops:', nx.number_of_selfloops(G))

def remove_self_loop(G):
    G.remove_edges_from(nx.selfloop_edges(G))

def expand_with_neighbours(G, node_list=None, default={}, **kwargs):
    # Get protein-protein interaction
    df = pd.read_csv('biogrid_ppi_graph.csv', index_col=0)
    ppi_graph = nx.from_pandas_edgelist(df, source='Interactor A', target='Interactor B')
    
    added_nodes = set()
    if node_list is None:
        node_list = list(G.nodes())
    for node in node_list:
        ### Check if the node must be expanded
        ### ugly implementation
        TO_EXPAND = True
        if kwargs != {}:
            for key in kwargs.keys():
                if G.nodes[node][key] not in kwargs[key]:
                    TO_EXPAND = False
        if not TO_EXPAND:
            continue 
        ####################
        # Get all the neighbours from the ppi graph
        try:
            neighbours = set(ppi_graph.neighbors(node))
        except nx.NetworkXError: # if the node is not in the graph
            continue
        # extract all the new nodes
        neighbours.difference_update(set(G.nodes()))
        G.add_nodes_from(neighbours, color='grey', **default)
        for n in neighbours:
            G.add_edge(node, n)
        added_nodes = added_nodes.union(neighbours)
    # Add edges between the new nodes - note: this also add self loops
    ppi_subgraph = ppi_graph.subgraph(added_nodes)
    G.add_edges_from(ppi_subgraph.edges())

def find_paths(G, source, depth_limit):
    paths = []
    to_expand = [[source, n] for n in G.neighbors(source)]
    while to_expand:
        path = to_expand.pop()
        # Check if last node in the path is a grey node
        if G.nodes[path[-1]]['color'] == 'grey':
            # If it is, keep expanding
            if len(path) < depth_limit + 1: # plus one because the also store the source node
                # Max depth not reached, can expand
                for neighbour in G.neighbors(path[-1]):
                    # Check each neighbour of the node to expand
                    # If the neighbout is not in the current path
                    if neighbour not in path:
                        path_copy = deepcopy(path)
                        path_copy.append(neighbour)
                        to_expand.append(path_copy)
        else:
            # Last node is blue, not expand the path anymore
            paths.append(deepcopy(path))

    min_length_paths = []
    blue_nodes = set([path[-1] for path in paths])
    for bn in blue_nodes:
        min_length_paths.append(min(filter(lambda x: bn == x[-1], paths), key=len))
    return min_length_paths

def get_main_connected_component(G):
    cc = sorted(nx.connected_components(G), key=len, reverse=True)[0]
    return G.subgraph(cc).copy()

def pruning_0(G):
    """
    Iteratively remove grey nodes which only have one neighbour
    """
    STOP_PRUNING = False
    while not STOP_PRUNING:
        STOP_PRUNING = True
        to_prune = []
        for n, color in G.nodes(data='color'):
            if color != 'grey':
                continue
            if len(list(G.neighbors(n))) == 1:
                to_prune.append(n)
        if len(to_prune) > 0:
            G.remove_nodes_from(to_prune)
            STOP_PRUNING = False

def D(S, T, node_index, shortest_path_matrix, diameter):
    min_dist = []
    T_gene_indexes = [node_index[t] for t in T]
    S_gene_indexes = [node_index[s] for s in S]
    for t in T_gene_indexes:
        distance = min(shortest_path_matrix[t, S_gene_indexes])
        min_dist.append(distance)
    score = np.mean(min_dist)
    return 1 - score/diameter

def exponential_decay_score(S, T, node_index, shortest_path_matrix):
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

def best_score(cs, disease_gene, target_gene, min, precomputed_scores, score_function, **kwargs):
    ncm = cs.to_node_community_map()
    ### Extrat the disease modules
    if ncm.get(disease_gene) is not None:
        # List of tuples (module_number, module)
        disease_modules = [(i, set(cs.communities[i])) for i in ncm[disease_gene]]
    else:
        # If the gene is not part of any module then [(None, {gene})]
        disease_modules = [(None, set([disease_gene]))]
    ### Extract the target modules
    if ncm.get(target_gene) is not None:
        target_modules = [(i, set(cs.communities[i])) for i in ncm[target_gene]]
    else:
        target_modules = [(None, set([target_gene]))]

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
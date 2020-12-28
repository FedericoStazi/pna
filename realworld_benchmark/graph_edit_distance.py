MAX_DIST = 4
'''
def graph_distance(a, b):
    dist = nx_sim.optimize_graph_edit_distance(a.to_networkx().to_undirected(),
                                              b.to_networkx().to_undirected(),
                                              upper_bound = MAX_DIST)
    return next(dist, MAX_DIST)
'''
def graph_distance(a, b):
    return abs(a.number_of_nodes() - b.number_of_nodes())

def embedding_distances(embeddings):
    n = len(scores)
    a = embeddings.repeat(n, 1)
    b = embeddings.unsqueeze(1).repeat(1, n, 1).flatten(end_dim=1)
    return torch.sum((a - b) ** 2, dim=1)
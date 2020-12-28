from scipy.optimize import linear_sum_assignment
import numpy as np

import os
import glob
import itertools
import torch

from itertools import chain
from scipy.spatial.distance import cdist
import networkx as nx


def graph_distance(a, b):
    d,_ = VanillaAED().ged(a.to_networkx().to_undirected(), b.to_networkx().to_undirected())
    return d

def embedding_distances(embeddings):
    n = len(embeddings)
    a = embeddings.repeat(n, 1)
    b = embeddings.unsqueeze(1).repeat(1, n, 1).flatten(end_dim=1)
    return torch.sum((a - b) ** 2, dim=1)

class GraphEditDistance(object):
    """
        An abstract class representing the Graph edit distance.
    """

    """
        Node edit operations
    """
    def node_substitution(self, g1, g2):
        raise NotImplementedError

    def node_insertion(self, g):
        raise NotImplementedError

    def node_deletion(self, g):
        raise NotImplementedError

    """
        Edge edit operations
    """
    def edge_substitution(self, g1, g2):
        raise NotImplementedError

    def edge_insertion(self, g):
        raise NotImplementedError

    def edge_deletion(self, g):
        raise NotImplementedError

    """
        Graph edit distance computation
    """
    def ged(self, g1, g2):
        raise NotImplementedError

class AproximatedEditDistance(GraphEditDistance):
    """
        An abstract class implementing the Graph edit distance aproximation proposed by Riesen and Bunke.
        The costs for nodes and edges must be defined by inheritance.
    """

    def edge_cost_matrix(self, g1, g2):
        cost_matrix = np.zeros([len(g1)+len(g2),len(g1)+len(g2)])

        # Insertion
        cost_matrix[len(g1):, 0:len(g2)] = np.inf
        np.fill_diagonal(cost_matrix[len(g1):, 0:len(g2)], self.edge_insertion(g1.values()))

        # Deletion
        cost_matrix[0:len(g1), len(g2):] = np.inf
        np.fill_diagonal(cost_matrix[0:len(g1), len(g2):], self.edge_deletion(g2.values()))

        # Substitution
        cost_matrix[0:len(g1), 0:len(g2)] = self.edge_substitution(g1.values(), g2.values())

        return cost_matrix

    """
        Aproximated graph edit distance for edges. The local structures are matched with this algorithm.
    """
    def edge_ed(self, g1, g2):

        # Compute cost matrix
        cost_matrix = self.edge_cost_matrix(g1, g2)

        # Munkres algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Graph edit distance
        dist = cost_matrix[row_ind, col_ind].sum()

        return dist

    def cost_matrix(self, g1, g2):
        cost_matrix = np.zeros([len(g1)+len(g2),len(g1)+len(g2)])

        # Insertion
        cost_matrix[len(g1):, 0:len(g2)] = np.inf
        np.fill_diagonal(cost_matrix[len(g1):, 0:len(g2)], np.concatenate((self.node_insertion(g1),self.edge_insertion(g1.edges.values()))))

        # Deletion
        cost_matrix[0:len(g1), len(g2):] = np.inf
        np.fill_diagonal(cost_matrix[0:len(g1), len(g2):], np.concatenate((self.node_insertion(g1),self.edge_insertion(g1.edges.values()))))

        # Substitution
        node_dist = self.node_substitution(g1, g2)

        i1 = 0
        for k1 in g1.nodes():
            i2 = 0
            for k2 in g2.nodes():
                node_dist[i1, i2] += self.edge_ed(g1[k1], g2[k2])
                i2 += 1
            i1 += 1

        cost_matrix[0:len(g1), 0:len(g2)] = node_dist
        return cost_matrix

    """
        Aproximated graph edit distance computation.
    """
    def ged(self, g1, g2):

        # Compute cost matrix
        cost_matrix = self.cost_matrix(g1, g2)

        # Munkres algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Graph edit distance
        dist = cost_matrix[row_ind, col_ind].sum()

        not_assign = np.invert((row_ind >= len(g1)) * (col_ind >= len(g2)))

        return dist, (row_ind[not_assign], col_ind[not_assign])

class VanillaAED(AproximatedEditDistance):
    """
        Vanilla Aproximated Edit distance, implements basic costs for substitution insertion and deletion.
    """

    def __init__(self, del_node = 0.5, ins_node = 0.5, del_edge = 0.25, ins_edge = 0.25, metric = "euclidean"):
        self.del_node = del_node
        self.ins_node = ins_node
        self.del_edge = del_edge
        self.ins_edge = ins_edge
        self.metric = metric

    """
        Node edit operations
    """
    def node_substitution(self, g1, g2):
        """
            Node substitution costs
            :param g1, g2: Graphs whose nodes are being substituted
            :return: Matrix with the substitution costs
        """
        values1 = [v for k, v in g1.nodes(data=True)]
        v1 = [list(chain.from_iterable(l.values())) for l in values1]

        values2 = [v for k, v in g2.nodes(data=True)]
        v2 = [list(chain.from_iterable(l.values())) for l in values2]

        node_dist = cdist(np.array(v1), np.array(v2), metric=self.metric)

        return node_dist

    def node_insertion(self, g):
        """
            Node Insertion costs
            :param g: Graphs whose nodes are being inserted
            :return: List with the insertion costs
        """
        values = [v for k, v in g.nodes(data=True)]
        return [self.ins_node]*len(values)

    def node_deletion(self, g):
        """
            Node Deletion costs
            :param g: Graphs whose nodes are being deleted
            :return: List with the deletion costs
        """
        values = [v for k, v in g.nodes(data=True)]
        return [self.del_node] * len(values)

    """
        Edge edit operations
    """
    def edge_substitution(self, g1, g2):
        """
            Edge Substitution costs
            :param g1, g2: Adjacency list for particular nodes.
            :return: List of edge deletion costs
        """
        edge_dist = cdist(np.array([list(l.values()) for l in g1]), np.array([list(l.values()) for l in g2]), metric=self.metric)
        return edge_dist

    def edge_insertion(self, g):
        """
            Edge insertion costs
            :param g: Adjacency list.
            :return: List of edge insertion costs
        """
        insert_edges = [len(e) for e in g]
        return np.array([self.ins_edge] * len(insert_edges)) * insert_edges

    def edge_deletion(self, g):
        """
            Edge Deletion costs
            :param g: Adjacency list.
            :return: List of edge deletion costs
        """
        delete_edges = [len(e) for e in g]
        return np.array([self.del_edge] * len(delete_edges)) * delete_edges
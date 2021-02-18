# MIT License
# Copyright (c) 2020 Vijay Prakash Dwivedi, Chaitanya K. Joshi, Thomas Laurent, Yoshua Bengio, Xavier Bresson


import random
import torch
import pickle
import torch.utils.data
import time
import numpy as np
import csv
import dgl
import math
from scipy import sparse as sp
import numpy as np
import networkx.algorithms.similarity as nx_sim
from graph_edit_distance import graph_distance

EPS = 1e-5
max_batch_distances = 10

# Can be removed?
class MoleculeDGL(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, num_graphs):
        pass

    def _prepare(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


class StructureAwareGraph(torch.utils.data.Dataset):
    # Create a StructureAwareGraph from a MoleculeDGL
    def __init__(self, molecule_dgl, features, label, max_graphs, precomputed_labels):
        self.data_dir = molecule_dgl.data_dir
        self.split = molecule_dgl.split
        max_graphs = max(min(max_graphs, molecule_dgl.num_graphs), 0)
        self.data = molecule_dgl.data[:max_graphs]
        self.num_graphs = self.n_samples = max_graphs
        if (self.split == "train"):
            self.num_graphs = self.n_samples = max_batch_distances * self.num_graphs
            for i in range(max_batch_distances - 1):
                data = molecule_dgl.data[:max_graphs]
                random.Random(i).shuffle(data)
                self.data.extend(data)
        self.graph_lists = []
        self._prepare(features, label)
        self.graph_labels = precomputed_labels[:len(self.graph_lists)]

    def _prepare(self, features, label):

        for molecule in self.data:
            #print("\rgraph %d out of %d" % (len(self.graph_lists), len(self.data)), end="")

            adj = molecule['bond_type']
            edge_list = (adj != 0).nonzero()  # converting adj matrix to edge_list

            edge_idxs_in_adj = edge_list.split(1, dim=1)
            edge_features = adj[edge_idxs_in_adj].reshape(-1).long()

            # Create the DGL Graph
            g = dgl.DGLGraph()
            g.add_nodes(molecule['num_atom'])

            for src, dst in edge_list:
                g.add_edges(src.item(), dst.item())
            g.edata['feat'] = edge_features

            # Set node features
            g.ndata['feat'] = torch.cuda.FloatTensor(
                [np.array(x) for x in np.array([f(g) for f in features]).transpose()])

            self.graph_lists.append(g)

        print()

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.graph_lists[idx], self.graph_labels[idx]

class MoleculeDataset(torch.utils.data.Dataset):

    def __init__(self, name, features, label, max_graphs, norm='none', verbose=True, normalization=False):
        """
            Loading SBM datasets
        """
        start = time.time()
        if verbose:
            print("[I] Loading dataset %s..." % (name))
        self.name = name
        data_dir = 'data/'
        with open(data_dir + name + '.pkl', "rb") as f:
            # Get precomputed labels from file
            self.precomputed_labels = []
            train_labels = list(map(float, open("data/precomputed_distances/train128.txt").read().split(",")))
            val_labels = list(map(float, open("data/precomputed_distances/val128.txt").read().split(",")))
            test_labels = list(map(float, open("data/precomputed_distances/test128.txt").read().split(",")))
            self.max_distance = max(
                max(train_labels),
                max(val_labels),
                max(test_labels)
            )
            # Load graphs
            f = pickle.load(f)
            self.train = StructureAwareGraph(f[0], features, label, max_graphs, train_labels)
            self.val = StructureAwareGraph(f[1], features, label, max_graphs, val_labels)
            self.test = StructureAwareGraph(f[2], features, label, max_graphs, test_labels)
            self.num_atom_type = f[3]
            self.num_bond_type = f[4]
        if verbose:
            print('train, test, val sizes :', len(self.train), len(self.test), len(self.val))
            print("[I] Finished loading.")
            print("[I] Data load time: {:.4f}s".format(time.time() - start))

        self.distances = {}
        self.total_graphs = (self.train.num_graphs
                             + self.val.num_graphs
                             + self.test.num_graphs)
        self.normalization = normalization

    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))

        # Normalization of labels
        if self.normalization:
            labels = [x / self.max_distance for x in labels]
        
        labels = torch.cuda.FloatTensor(labels)
        tab_sizes_n = [graphs[i].number_of_nodes() for i in range(len(graphs))]
        tab_snorm_n = [torch.FloatTensor(size, 1).fill_(1. / float(size)) for size in tab_sizes_n]
        snorm_n = torch.cat(tab_snorm_n).sqrt()
        tab_sizes_e = [graphs[i].number_of_edges() for i in range(len(graphs))]
        tab_snorm_e = [torch.FloatTensor(size, 1).fill_(1. / float(size)) for size in tab_sizes_e]
        snorm_e = torch.cat(tab_snorm_e).sqrt()
        batched_graph = dgl.batch(graphs)
        return batched_graph, labels, snorm_n, snorm_e

    def _add_self_loops(self):
        # function for adding self loops
        # this function will be called only if self_loop flag is True

        self.train.graph_lists = [self_loop(g) for g in self.train.graph_lists]
        self.val.graph_lists = [self_loop(g) for g in self.val.graph_lists]
        self.test.graph_lists = [self_loop(g) for g in self.test.graph_lists]

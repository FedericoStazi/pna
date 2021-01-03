# MIT License
# Copyright (c) 2020 Vijay Prakash Dwivedi, Chaitanya K. Joshi, Thomas Laurent, Yoshua Bengio, Xavier Bresson


import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import numpy as np
from graph_edit_distance import embedding_distances


def MAE(scores, targets, distance_function):
    print(len(scores), len(targets), len(embedding_distances(scores, distance_function)))
    MAE = F.l1_loss(torch.sqrt(embedding_distances(scores, distance_function)),
                    torch.sqrt(targets))
    return MAE

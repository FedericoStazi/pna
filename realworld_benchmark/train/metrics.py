# MIT License
# Copyright (c) 2020 Vijay Prakash Dwivedi, Chaitanya K. Joshi, Thomas Laurent, Yoshua Bengio, Xavier Bresson


import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import numpy as np
from graph_edit_distance import embedding_distances

def filter_positive(l):
    return filter(lambda x : x>0, l)


def MAE(scores, targets, distance_function):
    distances = filter_positive(embedding_distances(scores, distance_function))
    targets = filter_positive(targets)
    MAE = F.l1_loss(distances, targets)
    return MAE

def MSE(scores, targets, distance_function):
    distances = filter_positive(embedding_distances(scores, distance_function))
    targets = filter_positive(targets)
    MSE = F.mse_loss(distances, targets)
    return MSE

def MAPE(scores, targets, distance_function):
    distances = filter_positive(embedding_distances(scores, distance_function))
    targets = filter_positive(targets)
    MAPE = torch.mean(F.l1_loss(distances, targets, reduction='none') / torch.abs(distances))
    return MAPE
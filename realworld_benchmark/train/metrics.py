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
    distances = embedding_distances(scores, distance_function)
    valid = targets > 0
    MAE = F.l1_loss(distances[valid], targets[valid])
    return MAE

def MSE(scores, targets, distance_function):
    distances = embedding_distances(scores, distance_function)
    valid = targets > 0
    MSE = F.mse_loss(distances[valid], targets[valid])
    return MSE

def MAPE(scores, targets, distance_function):
    distances = embedding_distances(scores, distance_function)
    valid = targets > 0
    MAPE = torch.mean(F.l1_loss(distances[valid], targets[valid], reduction='none') / torch.abs(distances[valid]))
    return MAPE
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
    MAE = F.l1_loss(embedding_distances(scores, distance_function), targets)
    return MAE

def MSE(scores, targets, distance_function):
    MSE = F.mse_loss(embedding_distances(scores, distance_function), targets)
    return MSE

def MAPE(scores, targets, distance_function):
    MAPE = torch.mean(F.l1_loss(embedding_distances(scores, distance_function), targets, reduction='none') / torch.abs(targets))
    return MAPE

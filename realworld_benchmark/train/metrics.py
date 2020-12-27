# MIT License
# Copyright (c) 2020 Vijay Prakash Dwivedi, Chaitanya K. Joshi, Thomas Laurent, Yoshua Bengio, Xavier Bresson


import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import numpy as np


def MAE(scores, targets):
    n = len(scores)
    a = scores.repeat(n, 1)
    b = scores.unsqueeze(1).repeat(1, n, 1).flatten(end_dim=1)
    distances = torch.sum((a - b) ** 2, dim=1)
    print(distances, targets)
    MAE = F.l1_loss(distances, targets)
    return MAE

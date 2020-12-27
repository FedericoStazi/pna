# MIT License
# Copyright (c) 2020 Vijay Prakash Dwivedi, Chaitanya K. Joshi, Thomas Laurent, Yoshua Bengio, Xavier Bresson


import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import numpy as np


def MAE(scores, targets):
    distances = scores.unsqueeze(0).repeat(1, len(scores), 1)[0] - scores.repeat(len(scores), 1)
    distances = torch.sum(distances.pow(2), dim=1)
    MAE = F.l1_loss(distances, targets)
    return MAE

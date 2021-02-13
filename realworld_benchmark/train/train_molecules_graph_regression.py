# MIT License
# Copyright (c) 2020 Vijay Prakash Dwivedi, Chaitanya K. Joshi, Thomas Laurent, Yoshua Bengio, Xavier Bresson


"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math

from .metrics import MSE, MAE, MAPE

def train_epoch(model, optimizer, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    epoch_train_mse = 0
    epoch_train_mae = 0
    epoch_train_mape = 0
    nb_data = 0
    gpu_mem = 0
    for iter, (batch_graphs, batch_targets, batch_snorm_n, batch_snorm_e) in enumerate(data_loader):
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_snorm_e = batch_snorm_e.to(device)
        batch_targets = batch_targets.to(device)
        batch_snorm_n = batch_snorm_n.to(device)         # num x 1
        optimizer.zero_grad()
        batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_snorm_n, batch_snorm_e)
        loss = model.loss(batch_scores, batch_targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        mse = MSE(batch_scores, batch_targets, model.distance_function)
        mae = MAE(batch_scores, batch_targets, model.distance_function)
        mape = MAPE(batch_scores, batch_targets, model.distance_function)
        epoch_train_mse += mse
        epoch_train_mae += mae
        epoch_train_mape += mape
        #print("\ntrain ", batch_scores, batch_targets, mae)
        nb_data += batch_targets.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_mse /= (iter + 1)
    epoch_train_mae /= (iter + 1)
    epoch_train_mape /= (iter + 1)

    return epoch_loss, [epoch_train_mse, epoch_train_mae, epoch_train_mape], optimizer

def evaluate_network(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_mse = 0
    epoch_test_mae = 0
    epoch_test_mape = 0
    nb_data = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_targets, batch_snorm_n, batch_snorm_e) in enumerate(data_loader):
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_snorm_e = batch_snorm_e.to(device)
            batch_targets = batch_targets.to(device)
            batch_snorm_n = batch_snorm_n.to(device)
            
            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_snorm_n, batch_snorm_e)
            loss = model.loss(batch_scores, batch_targets)
            epoch_test_loss += loss.detach().item()
            mse = MSE(batch_scores, batch_targets, model.distance_function)
            mae = MAE(batch_scores, batch_targets, model.distance_function)
            mape = MAPE(batch_scores, batch_targets, model.distance_function)
            epoch_test_mse += mse
            epoch_test_mae += mae
            epoch_test_mape += mape
            #print("\nval ", batch_scores, batch_targets, mae)
            nb_data += batch_targets.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_mse /= (iter + 1)
        epoch_test_mae /= (iter + 1)
        epoch_test_mape /= (iter + 1)
        
    return epoch_test_loss, [epoch_test_mse, epoch_test_mae, epoch_test_mape]


def get_predictions(model, device, data_loader, epoch):
    model.eval()
    targets = []
    scores = []
    with torch.no_grad():
        for iter, (batch_graphs, batch_targets, batch_snorm_n, batch_snorm_e) in enumerate(data_loader):
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_snorm_e = batch_snorm_e.to(device)
            batch_targets = batch_targets.to(device)
            targets += batch_targets.flatten().toList()
            batch_snorm_n = batch_snorm_n.to(device)
            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_snorm_n, batch_snorm_e)
            scores += batch_scores.flatten().toList()
    return targets, scores

import collections

import numpy as np
import pandas as pd
import sklearn
import torch
from matplotlib import pyplot as plt


def train(optimizer, model, dataloader, metrics, loss_fn, prev_classification_loss, prev_clustering_loss):
    '''
    Train models in an epoch.
    '''
    model.train()
    total_loss = []
    preds = []
    ys = []
    for batch in dataloader:
        optimizer.zero_grad()
        beta = 0.1
        pred, mc_loss, o_loss, subgraph_mc_loss, ent_loss = model(*batch[:-1])
        preds.append(pred)
        ys.append(batch[-1])
        classification_loss = loss_fn(pred, batch[-1])
        clustering_loss = mc_loss + o_loss + subgraph_mc_loss + ent_loss
        final_loss = [1, 1]
        # if prev_classification_loss != 0:
        #     alpha1 = beta * (classification_loss - prev_classification_loss)
        #     alpha2 = beta * (clustering_loss - prev_clustering_loss)
        #     all_losses = torch.Tensor([alpha1, alpha2])
        #     final_loss = torch.softmax(all_losses, dim=-1)
        # print(f'Classification Loss: {classification_loss}')
        loss = final_loss[0] * classification_loss + final_loss[1] * clustering_loss
        loss.backward()
        total_loss.append(loss.detach().item())
        optimizer.step()
    pred = torch.cat(preds, dim=0)
    y = torch.cat(ys, dim=0)
    return metrics(pred.detach().cpu().numpy(), y.cpu().numpy()), sum(total_loss) / len(total_loss), classification_loss, clustering_loss


@torch.no_grad()
def test(f, model, dataloader, metrics, loss_fn):
    '''
    Test models either on validation dataset or test dataset.
    '''
    model.eval()
    preds = []
    ys = []
    for batch in dataloader:
        pred, mc_loss, o_loss, subgraph_mc_loss, ent_loss = model(*batch[:-1])
        preds.append(pred)
        ys.append(batch[-1])
    pred = torch.cat(preds, dim=0)
    y = torch.cat(ys, dim=0)
    return metrics(pred.cpu().numpy(), y.cpu().numpy()), loss_fn(pred, y) + (mc_loss + o_loss + subgraph_mc_loss + ent_loss)

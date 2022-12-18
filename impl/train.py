import collections

import numpy as np
import pandas as pd
import sklearn
import torch
from matplotlib import pyplot as plt


def train(optimizer, model, dataset, subgraph_assignment, metrics, loss_fn, prev_classification_loss, prev_clustering_loss):
    '''
    Train models in an epoch.
    '''
    preds = []
    ys = []
    model.train()
    optimizer.zero_grad()
    beta = 0.1
    pred, mc_loss, o_loss, subgraph_mc_loss, ent_loss = model(dataset.x, dataset.edge_index, dataset.edge_attr, dataset.pos, subgraph_assignment)
    classification_loss = loss_fn(pred, dataset.y)
    clustering_loss = mc_loss + o_loss + subgraph_mc_loss + ent_loss
    final_loss = [1, 1]
    if prev_classification_loss != 0:
        alpha1 = beta * (classification_loss - prev_classification_loss)
        alpha2 = beta * (clustering_loss - prev_clustering_loss)
        all_losses = torch.Tensor([alpha1, alpha2])
        final_loss = torch.softmax(all_losses, dim=-1)
    loss = final_loss[0] * classification_loss + final_loss[1] * clustering_loss
    loss.backward()
    optimizer.step()
    preds.append(pred)
    ys.append(dataset.y)
    pred = torch.cat(preds, dim=0)
    y = torch.cat(ys, dim=0)
    test = False
    if test:
        y_labels = y.cpu().numpy()
        new_preds = np.argmax(pred.detach().cpu().numpy(), 1)
        labels = new_preds - y_labels
        misclassification = np.count_nonzero(labels)
        nonzero_elements = np.nonzero(labels)
        print(f"Number of misclassifications: {misclassification}/{len(labels)}")
        predictions = new_preds[nonzero_elements]
        actual_labels = y_labels[nonzero_elements]
        commonly_mistaken = collections.Counter(actual_labels).most_common()
        commonly_mistaken_as = collections.Counter(predictions).most_common()
        print(f"Most commonly mistaken classes: {commonly_mistaken}")
        print(f"Most commonly mistaken as: {commonly_mistaken_as}")
        l = list(zip(actual_labels, predictions))
        l = sorted(l)
        common = collections.Counter(l).most_common()
        print(f"Most common tuples: {common}")
        confusion_matrix = sklearn.metrics.confusion_matrix(y_labels, new_preds)
        df_cm = pd.DataFrame(confusion_matrix, range(3), range(3))
        plt.figure(figsize=(10, 7))
        import seaborn as sn
        sn.set(font_scale=1.4)  # for label size
        # sn.color_palette("flare", as_cmap=True)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap="flare")  # font size
        plt.xlabel("Predicted", fontsize=20)
        plt.ylabel("Actual", fontsize=20)
        plt.show()
        # print(pred.cpu().numpy(), file=f)
    return metrics(pred.detach().cpu().numpy(), y.cpu().numpy()), loss, classification_loss, clustering_loss


@torch.no_grad()
def test(f, model, dataset, subgraph_assignment, metrics, loss_fn):
    '''
    Test models either on validation dataset or test dataset.
    '''
    model.eval()
    preds = []
    ys = []
    pred, mc_loss, o_loss, subgraph_mc_loss, ent_loss = model(dataset.x, dataset.edge_index, dataset.edge_attr, dataset.pos, subgraph_assignment)
    preds.append(pred)
    ys.append(dataset.y)
    pred = torch.cat(preds, dim=0)
    y = torch.cat(ys, dim=0)
    test = False
    if test:
        y_labels = y.cpu().numpy()
        new_preds = np.argmax(pred.cpu().numpy(), 1)
        labels = new_preds - y_labels
        misclassification = np.count_nonzero(labels)
        nonzero_elements = np.nonzero(labels)
        print(f"Number of misclassifications: {misclassification}/{len(labels)}")
        predictions = new_preds[nonzero_elements]
        actual_labels = y_labels[nonzero_elements]
        commonly_mistaken = collections.Counter(actual_labels).most_common()
        commonly_mistaken_as = collections.Counter(predictions).most_common()
        print(f"Most commonly mistaken classes: {commonly_mistaken}")
        print(f"Most commonly mistaken as: {commonly_mistaken_as}")
        l = list(zip(actual_labels, predictions))
        l = sorted(l)
        common = collections.Counter(l).most_common()
        print(f"Most common tuples: {common}")
        confusion_matrix = sklearn.metrics.confusion_matrix(y_labels, new_preds)
        df_cm = pd.DataFrame(confusion_matrix, range(3), range(3))
        plt.figure(figsize=(10,7))
        import seaborn as sn
        sn.set(font_scale=1.4)  # for label size
        # sn.color_palette("flare", as_cmap=True)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap="flare")  # font size
        plt.xlabel("Predicted", fontsize=20)
        plt.ylabel("Actual", fontsize=20)
        plt.show()
        # print(pred.cpu().numpy(), file=f)
    return metrics(pred.cpu().numpy(), y.cpu().numpy()), loss_fn(pred, y) + (mc_loss + o_loss + subgraph_mc_loss + ent_loss)

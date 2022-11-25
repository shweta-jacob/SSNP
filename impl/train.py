import collections

import numpy as np
import pandas as pd
import sklearn
import torch
from matplotlib import pyplot as plt
import seaborn as sn


def train(optimizer, model, dataset, subgraph_assignment, loss_fn, total_class_loss, total_clust_loss):
    '''
    Train models in an epoch.
    '''
    model.train()
    optimizer.zero_grad()
    beta = 0.1
    pred = model(dataset.x, dataset.edge_index, dataset.edge_attr, dataset.pos, subgraph_assignment)
    classification_loss = loss_fn(pred, dataset.y)
    # clustering_loss = mc_loss + o_loss + subgraph_mc_loss + ent_loss
    # final_loss = [1, 1]
    # if total_class_loss:
    #     alpha1 = beta * (classification_loss - total_class_loss[-1])
    #     alpha2 = beta * (clustering_loss - total_clust_loss[-1])
    #     all_losses = torch.Tensor([alpha1, alpha2])
    #     final_loss = torch.softmax(all_losses, dim=-1)
    loss = classification_loss
    loss.backward()
    optimizer.step()
    return loss, classification_loss


@torch.no_grad()
def test(f, model, dataset, subgraph_assignment, metrics, loss_fn):
    '''
    Test models either on validation dataset or test dataset.
    '''
    model.eval()
    preds = []
    ys = []
    pred = model(dataset.x, dataset.edge_index, dataset.edge_attr, dataset.pos, subgraph_assignment)
    preds.append(pred)
    ys.append(dataset.y)
    pred = torch.cat(preds, dim=0)
    y = torch.cat(ys, dim=0)
    # print(pred.cpu().numpy(), file=f)
    new_preds = np.argmax(pred.cpu().numpy(), 1)
    y_labels = y.cpu().numpy()
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
    sn.set(font_scale=1.4)  # for label size
    # sn.color_palette("flare", as_cmap=True)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap="flare")  # font size
    plt.xlabel("Predicted", fontsize=20)
    plt.ylabel("Actual", fontsize=20)
    plt.show()
    return metrics(pred.cpu().numpy(), y.cpu().numpy()), loss_fn(pred, y)

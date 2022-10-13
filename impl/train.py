import torch
import numpy as np
import networkx as nx
from matplotlib import pylab as pl
from torch_geometric.utils import to_networkx


def train(optimizer, model, dataloader, loss_fn):
    '''
    Train models in an epoch.
    '''
    model.train()
    total_loss = []
    for batch in dataloader:
        optimizer.zero_grad()
        pred = model(*batch[:-2], batch[6], id=0)
        loss = loss_fn(pred, batch[5])
        loss.backward()
        total_loss.append(loss.detach().item())
        optimizer.step()
    return sum(total_loss) / len(total_loss)


@torch.no_grad()
def test(model, dataloader, metrics, loss_fn, test=False, baseG=None):
    '''
    Test models either on validation dataset or test dataset.
    '''
    model.eval()
    preds = []
    ys = []
    for batch in dataloader:
        pred = model(*batch[:-2], batch[6])
        preds.append(pred)
        predicted_class = np.argmax(pred.cpu().numpy(), 1)
        ys.append(batch[5])
        if test:
            for pos in batch[3]:
                G = to_networkx(baseG, to_undirected=True)
                k = G.subgraph(pos.tolist())
                pos = nx.circular_layout(G)  # setting the positions with respect to G, not k.
                pl.figure()
                nx.draw_networkx(k, pos=pos)

                pl.show()

    pred = torch.cat(preds, dim=0)
    y = torch.cat(ys, dim=0)
    y_labels = y.cpu().numpy()
    # if test:
    #     new_preds = np.argmax(pred.cpu().numpy(), 1)
    #     labels = new_preds - y_labels
    #     misclassification = np.count_nonzero(labels)
    #     nonzero_elements = np.nonzero(labels)
    #     print(f"Number of misclassifications: {misclassification}/{len(labels)}")
    #     predictions = new_preds[nonzero_elements]
    #     actual_labels = y_labels[nonzero_elements]
    #     commonly_mistaken = collections.Counter(actual_labels).most_common()
    #     commonly_mistaken_as = collections.Counter(predictions).most_common()
    #     print(f"Most commonly mistaken classes: {commonly_mistaken}")
    #     print(f"Most commonly mistaken as: {commonly_mistaken_as}")
    #     l = list(zip(actual_labels, predictions))
    #     l = sorted(l)
    #     common = collections.Counter(l).most_common()
    #     print(f"Most common tuples: {common}")
    #     confusion_matrix = sklearn.metrics.confusion_matrix(y_labels, new_preds)
    #     df_cm = pd.DataFrame(confusion_matrix, range(6), range(6))
    #     plt.figure(figsize=(10,7))
    #     sn.set(font_scale=1.4)  # for label size
    #     # sn.color_palette("flare", as_cmap=True)
    #     sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap="flare")  # font size
    #     plt.xlabel("Predicted", fontsize=20)
    #     plt.ylabel("Actual", fontsize=20)
    #     plt.show()
    return metrics(pred.cpu().numpy(), y_labels), loss_fn(pred, y)

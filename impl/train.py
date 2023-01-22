import pandas as pd
import torch
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_emb_lookup(x, y):
    tsne = TSNE(n_components=2, verbose=1, n_iter=250)
    tsne_results = tsne.fit_transform(x.detach().numpy())

    df = pd.DataFrame()
    df["y"] = y
    df["comp-1"] = tsne_results[:, 0]
    df["comp-2"] = tsne_results[:, 1]

    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", 6),
                    data=df).set(title="Embedding Lookup T-SNE projection")

def plot_cont_labels(embs, y):
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    x1 = torch.index_select(torch.tensor(embs), 1, torch.tensor([0]))
    x2 = torch.index_select(torch.tensor(embs), 1, torch.tensor([1]))
    label = y
    colors = ['red', 'green', 'blue', 'purple', 'yellow', 'orange']

    fig = plt.figure(figsize=(8, 8))
    plt.scatter(x1.detach().numpy(), x2.detach().numpy(), c=label.detach().numpy(),
                cmap=matplotlib.colors.ListedColormap(colors))

    cb = plt.colorbar()
    loc = np.arange(0, max(label), max(label) / float(len(colors)))
    cb.set_ticks(loc)
    cb.set_ticklabels(colors)
    plt.show()


def train(optimizer, model, dataloader, metrics, loss_fn):
    '''
    Train models in an epoch.
    '''
    model.train()
    total_loss = []
    preds = []
    ys = []
    embs = []
    init_embs = []
    for batch in dataloader:
        optimizer.zero_grad()
        pred, init_emb, emb, mc_loss, o_loss, ent_loss = model(*batch[:-1])
        init_embs.append(init_emb)
        embs.append(emb)
        preds.append(pred)
        ys.append(batch[-1])
        classification_loss = loss_fn(pred, batch[-1])
        clustering_loss = mc_loss + o_loss + ent_loss
        loss = classification_loss + clustering_loss
        loss.backward()
        total_loss.append(loss.detach().item())
        optimizer.step()
    pred = torch.cat(preds, dim=0)
    y = torch.cat(ys, dim=0)
    embs = torch.cat(embs, dim=0)
    init_embs = torch.cat(init_embs, dim=0)
    # plot_emb_lookup(init_embs, y)
    # plot_emb_lookup(embs, y)
    # pca = PCA(n_components=2, svd_solver='full')
    # init_embs = pca.fit_transform(init_embs.detach().numpy())
    # plot_cont_labels(init_embs, y)
    plot_cont_labels(embs.detach().numpy(), y)
    return metrics(pred.detach().cpu().numpy(), y.cpu().numpy()), sum(total_loss) / len(
        total_loss)


@torch.no_grad()
def test(model, dataloader, metrics, loss_fn):
    '''
    Test models either on validation dataset or test dataset.
    '''
    model.eval()
    preds = []
    ys = []
    total_loss = []
    for batch in dataloader:
        pred, init_emb, emb, mc_loss, o_loss, ent_loss = model(*batch[:-1])
        preds.append(pred)
        ys.append(batch[-1])
        classification_loss = loss_fn(pred, batch[-1])
        clustering_loss = mc_loss + o_loss + ent_loss
        loss = classification_loss + clustering_loss
        total_loss.append(loss.detach().item())
    pred = torch.cat(preds, dim=0)
    y = torch.cat(ys, dim=0)
    return metrics(pred.cpu().numpy(), y.cpu().numpy()), sum(total_loss) / len(
        total_loss)

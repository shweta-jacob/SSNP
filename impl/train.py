import pandas as pd
import torch
import warnings
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

warnings.filterwarnings(action="ignore")

def plot_embs_tsne(x, y, title):
    tsne = TSNE(n_components=2, verbose=0, n_iter=250, perplexity=2)
    x = x.detach().numpy()
    tsne_results = tsne.fit_transform(x)

    df = pd.DataFrame()
    df["y"] = y
    df["comp-1"] = tsne_results[:, 0]
    df["comp-2"] = tsne_results[:, 1]

    print(f"{title} SIL: {silhouette_score(x, y):.4f}")

    # sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
    #                 palette=sns.color_palette("hls", 4),
    #                 data=df).set(title=f"{title} SIL: {silhouette_score(x, y):.4f}")
    # plt.show()


def train(optimizer, model, dataloader, metrics, loss_fn, epoch):
    '''
    Train models in an epoch.
    '''
    model.train()
    total_loss = []
    preds = []
    ys = []
    all_cont_labels = []
    all_subg_embs = []
    s = None
    for batch in dataloader:
        optimizer.zero_grad()
        pred, subg_embs, cont_labels, mc_loss, o_loss, ent_loss, s = model(*batch[:-1])
        all_subg_embs.append(subg_embs)
        all_cont_labels.append(cont_labels)
        preds.append(pred)
        ys.append(batch[-1])
        beta = 0
        classification_loss = loss_fn(pred, batch[-1])
        clustering_loss = mc_loss + o_loss + ent_loss
        loss = classification_loss + beta * clustering_loss
        loss.backward()
        total_loss.append(loss.detach().item())
        optimizer.step()
    pred = torch.cat(preds, dim=0)
    y = torch.cat(ys, dim=0)
    if epoch == 49:
        all_cont_labels = torch.cat(all_cont_labels, dim=0)
        all_subg_embs = torch.cat(all_subg_embs, dim=0)
        plot_embs_tsne(all_subg_embs, y, "Embedding T-SNE projection")
        plot_embs_tsne(all_cont_labels, y, "Contribution Label T-SNE projection")
    return metrics(pred.detach().cpu().numpy(), y.cpu().numpy()), sum(total_loss) / len(
        total_loss), s


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
        pred, init_emb, emb, mc_loss, o_loss, ent_loss, s = model(*batch[:-1])
        preds.append(pred)
        ys.append(batch[-1])
        classification_loss = loss_fn(pred, batch[-1])
        clustering_loss = mc_loss + o_loss + ent_loss
        beta = 0
        loss = classification_loss + beta * clustering_loss
        total_loss.append(loss.detach().item())
    pred = torch.cat(preds, dim=0)
    y = torch.cat(ys, dim=0)
    return metrics(pred.cpu().numpy(), y.cpu().numpy()), sum(total_loss) / len(
        total_loss)

import torch


def train(optimizer, model, dataset, subgraph_assignment, loss_fn, total_class_loss, total_clust_loss):
    '''
    Train models in an epoch.
    '''
    model.train()
    optimizer.zero_grad()
    beta = 0.1
    pred, mc_loss, o_loss, subgraph_mc_loss, ent_loss = model(dataset.x, dataset.edge_index, dataset.edge_attr, dataset.pos, subgraph_assignment)
    classification_loss = loss_fn(pred, dataset.y[0:5])
    clustering_loss = mc_loss + o_loss + subgraph_mc_loss + ent_loss
    final_loss = [1, 1]
    if total_class_loss:
        alpha1 = beta * (classification_loss - total_class_loss[-1])
        alpha2 = beta * (clustering_loss - total_clust_loss[-1])
        all_losses = torch.Tensor([alpha1, alpha2])
        final_loss = torch.softmax(all_losses, dim=-1)
    loss = final_loss[0] * classification_loss + final_loss[1] * clustering_loss
    loss.backward()
    optimizer.step()
    return loss, classification_loss, clustering_loss


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
    ys.append(dataset.y[0:5])
    pred = torch.cat(preds, dim=0)
    y = torch.cat(ys, dim=0)
    # print(pred.cpu().numpy(), file=f)
    return metrics(pred.cpu().numpy(), y.cpu().numpy()), loss_fn(pred, y) + (mc_loss + o_loss + subgraph_mc_loss + ent_loss)

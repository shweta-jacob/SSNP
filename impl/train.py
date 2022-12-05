import torch


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
    # print(pred.cpu().numpy(), file=f)
    return metrics(pred.cpu().numpy(), y.cpu().numpy()), loss_fn(pred, y) + (mc_loss + o_loss + subgraph_mc_loss + ent_loss)

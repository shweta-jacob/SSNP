import torch


def train(optimizer, model, dataset, subgraph_assignment, loss_fn):
    '''
    Train models in an epoch.
    '''
    model.train()
    optimizer.zero_grad()
    pred, mc_loss, o_loss = model(dataset.x, dataset.edge_index, dataset.edge_attr, dataset.pos, subgraph_assignment)
    loss = loss_fn(pred, dataset.y) + 0.5 * (mc_loss + o_loss)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test(model, dataset, subgraph_assignment, metrics, loss_fn):
    '''
    Test models either on validation dataset or test dataset.
    '''
    model.eval()
    preds = []
    ys = []
    pred, mc_loss, o_loss = model(dataset.x, dataset.edge_index, dataset.edge_attr, dataset.pos, subgraph_assignment)
    preds.append(pred)
    ys.append(dataset.y)
    pred = torch.cat(preds, dim=0)
    y = torch.cat(ys, dim=0)
    return metrics(pred.cpu().numpy(), y.cpu().numpy()), loss_fn(pred, y) + (mc_loss + o_loss)

import torch


def train(optimizer, model, dataset, loss_fn, device):
    '''
    Train models in an epoch.
    '''
    model.train()
    total_loss = []
    optimizer.zero_grad()
    pred = model(dataset.x, dataset.edge_index, dataset.edge_attr, dataset.pos, device=device, id=0)
    loss = loss_fn(pred, dataset.y)
    loss.backward()
    total_loss.append(loss.detach().item())
    optimizer.step()
    return sum(total_loss)/ len(total_loss)


@torch.no_grad()
def test(model, dataset, metrics, loss_fn, device):
    '''
    Test models either on validation dataset or test dataset.
    '''
    model.eval()
    preds = []
    ys = []
    pred = model(dataset.x, dataset.edge_index, dataset.edge_attr, dataset.pos, device=device)
    preds.append(pred)
    ys.append(dataset.y)
    pred = torch.cat(preds, dim=0)
    y = torch.cat(ys, dim=0)
    return metrics(pred.cpu().numpy(), y.cpu().numpy()), loss_fn(pred, y)

import torch
from tqdm import tqdm


def train(optimizer, model, dataloader, metrics, loss_fn):
    '''
    Train models in an epoch.
    '''
    model.train()
    total_loss = []
    ys = []
    preds = []
    pbar = tqdm(dataloader, ncols=70, disable=True)
    for data in pbar:
        optimizer.zero_grad()
        pred = model(data.num_nodes, data.z, data.edge_index, data.batch, data.x, data.edge_weight, data.node_id)
        loss = loss_fn(pred, data.y)
        preds.append(pred)
        ys.append(data.y)
        loss.backward()
        total_loss.append(loss.detach().item())
        optimizer.step()
    pred = torch.cat(preds, dim=0)
    y = torch.cat(ys, dim=0)
    return metrics(pred.detach().cpu().numpy(), y.detach().cpu().numpy()), sum(total_loss) / len(total_loss)


@torch.no_grad()
def test(model, dataloader, metrics, loss_fn):
    '''
    Test models either on validation dataset or test dataset.
    '''
    model.eval()
    preds = []
    ys = []
    pbar = tqdm(dataloader, ncols=70, disable=True)
    for data in pbar:
        pred = model(data.num_nodes, data.z, data.edge_index, data.batch, data.x, data.edge_weight, data.node_id)
        preds.append(pred)
        ys.append(data.y)
    pred = torch.cat(preds, dim=0)
    y = torch.cat(ys, dim=0)
    return metrics(pred.cpu().numpy(), y.cpu().numpy()), loss_fn(pred, y)

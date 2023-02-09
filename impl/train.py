import torch
from tqdm import tqdm


def train(optimizer, model, dataloader, metrics, loss_fn, device):
    '''
    Train models in an epoch.
    '''
    model.train()
    total_loss = []
    ys = []
    preds = []
    pbar = tqdm(dataloader, ncols=70, disable=True)
    for data in pbar:
        data = data.to(device)
        optimizer.zero_grad()
        pred = model(data.x.to(device), data.batch)
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
def test(model, dataloader, metrics, loss_fn, device):
    '''
    Test models either on validation dataset or test dataset.
    '''
    model.eval()
    preds = []
    ys = []
    pbar = tqdm(dataloader, ncols=70, disable=True)
    for data in pbar:
        data = data.to(device)
        pred = model(data.x.to(device), data.batch)
        preds.append(pred)
        ys.append(data.y)
    pred = torch.cat(preds, dim=0)
    y = torch.cat(ys, dim=0)
    return metrics(pred.cpu().numpy(), y.cpu().numpy()), loss_fn(pred, y)

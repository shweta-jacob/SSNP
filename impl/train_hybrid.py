import random
import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi gpu


def train(optimizer, model, dataloader, metrics, loss_fn, device, row, col, run, epoch):
    '''
    Train models in an epoch.
    '''
    model.train()
    total_loss = []
    ys = []
    preds = []
    row = row.to(device)
    col = col.to(device)
    for batch in dataloader:
        optimizer.zero_grad()
        pred = model(*batch[:-1], row=row, col=col, device=device, id=0)
        loss = loss_fn(pred, batch[-1])
        preds.append(pred)
        ys.append(batch[-1])
        loss.backward()
        total_loss.append(loss.detach().item())
        optimizer.step()
    pred = torch.cat(preds, dim=0)
    y = torch.cat(ys, dim=0)
    return metrics(pred.detach().cpu().numpy(), y.detach().cpu().numpy()), sum(total_loss) / len(total_loss)


@torch.no_grad()
def test(model, dataloader, metrics, loss_fn, device, row, col, run, epoch):
    '''
    Test models either on validation dataset or test dataset.
    '''
    model.eval()
    preds = []
    ys = []
    row = row.to(device)
    col = col.to(device)
    for batch in dataloader:
        pred = model(*batch[:-1], row=row, col=col, device=device)
        preds.append(pred)
        ys.append(batch[-1])
    pred = torch.cat(preds, dim=0)
    y = torch.cat(ys, dim=0)
    return metrics(pred.cpu().numpy(), y.cpu().numpy()), loss_fn(pred, y)

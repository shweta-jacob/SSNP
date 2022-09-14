import torch


def train(optimizer, model, dataloader, sub_loader, loss_fn, device):
    '''
    Train models in an epoch.
    '''
    model.train()

    iterator1 = iter(sub_loader)
    # iterator2 = iter(comp_loader)
    total_loss = []
    for batch in dataloader:
        optimizer.zero_grad()
        first = next(iterator1)
        # second = next(iterator2)
        pred = model(*batch[:-1], *first[:-1], id=0)
        loss = loss_fn(pred, batch[-1])
        loss.backward()
        total_loss.append(loss.detach().item())
        optimizer.step()
    return sum(total_loss) / len(total_loss)


@torch.no_grad()
def test(model, dataloader, sub_loader, metrics, loss_fn, device):
    '''
    Test models either on validation dataset or test dataset.
    '''
    model.eval()
    preds = []
    ys = []
    iterator1 = iter(sub_loader)
    # iterator2 = iter(comp_loader)

    for batch in dataloader:
        first = next(iterator1)
        # second = next(iterator2)
        pred = model(*batch[:-1], *first[:-1])
        preds.append(pred)
        ys.append(batch[-1])
    pred = torch.cat(preds, dim=0)
    y = torch.cat(ys, dim=0)
    return metrics(pred.cpu().numpy(), y.cpu().numpy()), loss_fn(pred, y)

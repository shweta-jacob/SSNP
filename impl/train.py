import collections
import numpy as np
import torch


def train(optimizer, model, dataloader, loss_fn):
    '''
    Train models in an epoch.
    '''
    model.train()
    total_loss = []
    for batch in dataloader:
        optimizer.zero_grad()
        pred = model(*batch[:-1], id=0)
        loss = loss_fn(pred, batch[-1])
        loss.backward()
        total_loss.append(loss.detach().item())
        optimizer.step()
    return sum(total_loss) / len(total_loss)


@torch.no_grad()
def test(model, dataloader, metrics, loss_fn, test=False):
    '''
    Test models either on validation dataset or test dataset.
    '''
    model.eval()
    preds = []
    ys = []
    for batch in dataloader:
        pred = model(*batch[:-1])
        preds.append(pred)
        ys.append(batch[-1])
    pred = torch.cat(preds, dim=0)
    y = torch.cat(ys, dim=0)
    y_labels = y.cpu().numpy()
    if test:
        new_preds = np.argmax(pred.cpu().numpy(), 1)
        labels = new_preds - y_labels
        misclassification = np.count_nonzero(labels)
        nonzero_elements = np.nonzero(labels)
        print(f"Number of misclassifications: {misclassification}/{len(labels)}")
        predictions = new_preds[nonzero_elements]
        actual_labels = y_labels[nonzero_elements]
        commonly_mistaken = collections.Counter(actual_labels).most_common()
        commonly_mistaken_as = collections.Counter(predictions).most_common()
        print(f"Most commonly mistaken classes: {commonly_mistaken}")
        print(f"Most commonly mistaken as: {commonly_mistaken_as}")
        l = list(zip(actual_labels, predictions))
        l = sorted(l)
        common = collections.Counter(l).most_common()
        print(f"Most common tuples: {common}")
    return metrics(pred.cpu().numpy(), y_labels), loss_fn(pred, y)

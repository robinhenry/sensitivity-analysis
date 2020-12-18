import torch
import torch.nn as nn
import numpy as np
import copy


def train(model, train_data, val_data, lr=1e-3, epochs=20, l2=0.):

    # Define loss function and optimizer.
    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

    # Keep track of the best model and validation loss.
    best_model = None
    best_val_loss = None

    # Train neural net for `epochs` epochs.
    for epoch in range(epochs):

        model = model.train(True)

        for x, pq, y in train_data:
            optimizer.zero_grad()
            out = model(x, pq)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        if epoch % 1 == 0:
            val_loss = _validate(model, val_data, criterion)

            print(f'Epoch {epoch+1}, val loss: {val_loss:2.10f}')

            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(model)

    return best_model, best_val_loss


def _validate(model, val_data, criterion):
    """ Compute the validation loss. """

    loss = 0.
    model = model.train(False)

    N_val = len(val_data)
    with torch.no_grad():
        for x, pq, y in val_data:
            out = model(x, pq)
            l = criterion(out, y)
            loss += l.detach().item() / N_val

    return loss

def checkpoint_filename(dataset, sc, net_type, model_type, x_i, seed):
    s = f'trained_{net_type}_{dataset}_{sc}_{model_type}_x{x_i}_rs{seed}.pt'
    return s
import torch
from torch import nn as nn
import copy


def train(model, train_data, val_data, lr, epoch_max, l2):


    # Define loss function and optimizer.
    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

    # Empty lists to store training losses.
    avg_train_loss = []

    best_model = None
    best_loss = None
    best_epoch = None

    # Train neural net for `epochs` epochs.
    N_train = len(train_data)
    for epoch in range(epoch_max):

        model = model.train(True)
        epoch_loss = 0.

        y_true, y_pred = [], []

        for x, pq, y in train_data:
            optimizer.zero_grad()
            out = model(x, pq)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach().item() / N_train

            y_true.append(y.detach().numpy())
            y_pred.append(out.detach().numpy())

        # Compute validation loss.
        val_loss = validate(model, val_data, criterion)

        # Print training loss.
        print(f'Epoch {epoch}: training loss = {epoch_loss:.5e}, val loss = {val_loss:.5e}')

        # Store the training loss.
        avg_train_loss.append(epoch_loss)

        if best_loss is None or val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model)
            best_epoch = epoch

    return best_loss, best_model, best_epoch


def validate(model, val_data, criterion):
    model = model.train(False)
    val_loss = 0.

    N_val = len(val_data)
    for x, pq, y in val_data:
        out = model(x, pq)
        loss = criterion(out, y)

        val_loss += loss.detach().item() / N_val

    return val_loss

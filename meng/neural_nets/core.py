import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def train(model, train_dataloader, lr=1e-3, epochs=20, l2=0., plot=False):

    # Define loss function and optimizer.
    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

    # Empty lists to store training losses.
    avg_train_loss = []

    if plot:
        n_plots = int(epochs / 5)
        fig, axs = plt.subplots(n_plots, 1, sharex=True, figsize=(10, 2*n_plots))
        if not isinstance(axs, np.ndarray):
            axs = [axs]

    # Train neural net for `epochs` epochs.
    N_train = len(train_dataloader)
    for epoch in range(epochs):

        model = model.train(True)
        epoch_loss = 0.

        y_true, y_pred = [], []

        for x, pq, y in train_dataloader:
            optimizer.zero_grad()
            out = model(x, pq)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach().item() / N_train

            y_true.append(y.detach().numpy())
            y_pred.append(out.detach().numpy())

        # Print training loss.
        print(f'Epoch {epoch}: training loss = {epoch_loss:.5e}')

        if epoch % 5 == 0 and plot :
            e = int(epoch / 5)
            T = 200
            y_true = np.hstack(y_true)
            y_pred = np.hstack(y_pred)
            axs[e].plot(y_true[:T], label='True')
            axs[e].plot(y_pred[:T], label='Predicted')
            axs[e].legend(loc='upper right')

        # Store the training loss.
        avg_train_loss.append(epoch_loss)

    return avg_train_loss


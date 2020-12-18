import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def train(model, train_data, epochs, lr=1e-3, batch_size=64):
    """
    Train an LSTM to estimate sensitivity coefficients.

    Parameters
    ----------
    model : nn.Module
        A Pytorch model.
    train_seq : list of tuple
        The list of training sequences. See `create_inout_sequences`.
    epochs : int
        The number of training epochs.
    lr : float, optional
        The learning rate.
    """

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train LSTM.
    model = model.train(True)

    for i in range(epochs):
        loss = 0.

        for seq, pq, labels in tqdm(train_data):
            optimizer.zero_grad()

            y_pred, _ = model(seq, pq)

            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

            loss += single_loss.detach().item()

        print(f'Epoch: {i:3}, loss: {loss:10.8f}')


def predict(model, test_data, batch_size=64):
    """ Predict sensitivity coefficients and dependent variables using a
    trained LSTM.

    Parameters
    ----------
    model : nn.Module
        The trained LSTM network.
    test_data : list of tuple
        The list of input sequences. See `create_inout_sequences`.

    Returns
    -------
    coeff_predicted : (2N, T) array_like
        The predicted coefficients.
    x_predicted : (T,) array_like
        The dependent variable estimation.
    x_true : (T,) array_like
        The dependent variable target values (i.e., noisy measurements).
    """
    coeffs_predicted = []
    v_predicted = []
    v_true = []

    model = model.train(False)

    with torch.no_grad():
        for seq, pq, labels in test_data:
            v_pred, coeff_pred = model(seq, pq)

            coeffs_predicted.append(coeff_pred.numpy())
            v_predicted.append(v_pred.numpy())
            v_true.append(labels.numpy())

    coeffs_predicted = np.vstack(coeffs_predicted).T
    v_predicted = np.hstack(v_predicted)
    v_true = np.hstack(v_true)

    return coeffs_predicted, v_predicted, v_true



if __name__ == '__main__':
    pass

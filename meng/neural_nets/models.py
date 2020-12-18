import torch
import torch.nn as nn
import numpy as np


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def predict(self, dataloader):

        ts = []
        S, y_pred, y_true = [], [], []

        self.train(False)
        with torch.no_grad():
            for i, (x, pq, y) in enumerate(dataloader):
                c = self.predict_coefficients(x).numpy()
                out = self.forward(x, pq).numpy()

                S.append(c)
                y_pred.append(out)
                y_true.append(y.numpy())
                ts.append(self.k + i)

        S = np.vstack(S).T
        y_pred = np.hstack(y_pred)
        y_true = np.hstack(y_true)

        return S, y_pred, y_true

    def forward(self, x, pq):
        raise NotImplementedError()


class FeedForward(BaseModel):

    def __init__(self, in_shape, hidden_shapes, sc_matrix_shape, k,
                 activation=nn.ReLU):
        """
        Parameters
        ----------
        in_shape : int
            The size of input vectors.
        hidden_shapes : list of int
            The number of units for each hidden layer.
        sc_matrix_shape : tuple of int
            The shape of the sensitivity coefficient matrix to output.
        k : int
            The number of previous time steps included in the input vectors.
        activation : function
            The nonlinear activation function to apply before hidden layers.
        """

        super().__init__()

        self.k = k
        if np.isscalar(sc_matrix_shape):
            self.sc_matrix_shape = [sc_matrix_shape]
        else:
            self.sc_matrix_shape = sc_matrix_shape

        out_shape = np.prod(sc_matrix_shape).item()

        # Create a feed forward fully-connected neural net.
        shapes = [in_shape] + hidden_shapes
        layers = []
        for i in range(len(shapes) - 1):
            layers.append(nn.Linear(shapes[i], shapes[i + 1]))
            layers.append(activation())
        layers.append(nn.Linear(shapes[-1], out_shape))

        self.net = nn.Sequential(*layers).float()

    def forward(self, x, pq):
        """
        Predict the next dependent variable measurement.

        Parameters
        ----------
        x : torch.Tensor
            The (B, in_shape) input tensor that contains the
            recent history of measurements.
        pq : torch.Tensor
            The (B, N) tensor of independent variable measurement to use in the
            prediction of the next dependent variable measurement.

        Returns
        -------
        torch.Tensor
            The (B,) tensor of estimated dependent variable measurements.
        """

        coefficients = self.predict_coefficients(x)
        out = torch.sum(coefficients * pq, dim=-1)

        return out

    def predict_coefficients(self, x):
        """
        Predict the sensitivity coefficients.

        Parameters
        ----------
        x : torch.Tensor
            The (B, in_shape) input tensor that contains the recent history of
            measurements.

        Returns
        -------
        torch.Tensor
            The (B, N) vector of sensitivity coefficients estimated.
        """

        out = self.net(x)
        out = out.view(([-1] + list(self.sc_matrix_shape)))

        return out
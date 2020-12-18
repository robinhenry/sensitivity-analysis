import torch
from torch import nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, batch_size):
        super().__init__()

        self.hidden_layer_size = hidden_layer_size

        # Used to predict coefficients.
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = self.init_hidden(batch_size)

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_layer_size),
                torch.zeros(1, batch_size, self.hidden_layer_size),)

    def forward(self, input_seq, pq):

        # Re-initialize hidden cell.
        self.hidden_cell = self.init_hidden(input_seq.shape[0])

        # Predict coefficients.
        coefficients = self._predict_coefficients(input_seq)

        # Use coefficients to predict dependent variables.
        predictions = torch.sum(coefficients * pq, dim=-1)

        return predictions, coefficients

    def _predict_coefficients(self, input_seq):

        # Predict sensitivity coefficients.
        # input_seq = input_seq.view(len(input_seq), 1, -1)
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        coefficients = self.linear(lstm_out)
        coefficients = coefficients[:, -1, :]

        return coefficients

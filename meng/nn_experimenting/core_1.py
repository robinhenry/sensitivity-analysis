import numpy as np
from torch.utils.data import Dataset, DataLoader


class DifferentialOutputDataset(Dataset):

    def  __init__(self, X, PQ, y, x_i, k):
        """
        Parameters
        ----------
        X : (M, T) array_like
            The input measurements (e.g., [|X|, P, Q]), possibly normalized.
        PQ : (N, T) array_like
            The independent variable measurements (e.g., PQ injections).
        y : (N, T) array_like
            The dependent variable measurements to be approximated.
        x_i : int
            The index of the dependent variable for which to learn the sensitivity
            coefficients (e.g., bus index). It is 1-indexed.
        k : int
            The number of previous timesteps fed into the neural network.
        """

        self.k = k
        self.X = X.astype(np.float32)
        self.PQ = PQ.astype(np.float32)
        self.y = y[x_i - 1].astype(np.float32)
        self.M = X.shape[0]
        self.N, self.T = PQ.shape

    def __len__(self):
        # T - k
        return self.T - self.k

    def __getitem__(self, item):
        """
        Return the `item` th data point.

        Parameters
        ----------
        item : int
            The index of the data point to return in [0, T-k-1].

        Returns
        -------
        x : ((M + N) * k,) array_like
            The measurements over the window [item, item + k - 1] flattened in a
            1D array.
        pq : (N,) array_like
            The vector of differential power injections at time `item + k`,
            which gets used to estimate (predict) the dependent variable
            measurement (see `y` below) at time `item + k`.
        y : float
            The dependent variable deviaiton measurement to be estimated at
            time `item + k`.
        """

        # Neural net input vector.
        x = self.X[:, item: item + self.k].flatten()
        # pq = self.PQ[:, item: item + self.k].flatten()
        # x = np.hstack((x, pq))
        assert x.shape == (self.M * self.k,)

        # Output vector to be learned.
        y = self.y[item + self.k] - self.y[item + self.k - 1]
        assert np.shape(y) == ()  # scalar

        # PQ matrix used inn the reconstruction of the output.
        pq = self.PQ[:, item + self.k] - self.PQ[:, item + self.k - 1]
        assert pq.shape == (self.N,)

        return x, pq, y


class SequenceDataset(Dataset):

    def __init__(self, X, PQ, y, x_i, time_window):

        self.k = time_window
        self.X = X.astype(np.float32)
        self.PQ = PQ.astype(np.float32)
        self.y = y[x_i - 1].astype(np.float32)
        self.T = X.shape[1]

    def __len__(self):
        return self.T - self.k

    def __getitem__(self, t):

        # Input sequence.
        x_seq = self.X[:, t: t + self.k]  # voltage measurements
        # pq_seq = self.PQ[:, t: t + self.k]  # power inj measurements
        # x_seq = np.vstack((pq_seq, x_seq))      # (tw, 3N)
        x_seq = x_seq.T

        # PQ used in dependent variable estimations.
        pq = self.PQ[:, t + self.k] - self.PQ[:, t + self.k - 1]  # (2N,)

        # Output (dependent variable).
        label = self.y[t + self.k] - self.y[t + self.k - 1] # (1,)

        return x_seq, pq, label


def build_lstm_dataloader(X, PQ, y, x_i, time_window, batch_size=64):
    dataset = SequenceDataset(X, PQ, y, x_i, time_window)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def build_training_dataloader(X, PQ, y, x_i, k, batch_size=64):
    dataset = DifferentialOutputDataset(X, PQ, y, x_i, k)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def build_testing_dataloader(X, PQ, y, x_i, k, batch_size=64):
    dataset = DifferentialOutputDataset(X, PQ, y, x_i, k)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)



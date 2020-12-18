import numpy as np
from torch.utils.data import Dataset, DataLoader


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


def build_dataloader(X, PQ, y, x_i, time_window, batch_size=64):
    dataset = SequenceDataset(X, PQ, y, x_i, time_window)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

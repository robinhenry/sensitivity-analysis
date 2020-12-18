import matplotlib.pyplot as plt

def single_plot(x, ys, labels=None, xlabel=None, ylabel=None, title=None,
                figsize=(10, 3), ylim=(None, None), ax=None, legend_loc='upper right'):
    """
    Make a single plot of 1 or more curves.

    Parameters
    ----------
    x : array_like
        The xtick values.
    ys : array_like or list of array_like
        The curves to plot.
    labels : list of str
        The label of each curve.
    xlabel : str
        The xlabel.
    ylabel : str
        The ylabel.

    Returns
    -------
    ax
    """

    # Take care of labels.
    if labels is None:
        labels = [None] * (len(ys) + 1)
    elif not isinstance(labels, list):
        labels = [labels]

    # Special case where only 1 curve is provided.
    if not isinstance(ys, list):
        ys = [ys]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = None

    for y, label in zip(ys, labels):
        ax.plot(x, y, label=label)
    ax.legend(loc=legend_loc)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(ylim)

    return fig, ax


def shaded_plot(xs, ys, labels=None, xlabel=None, ylabel=None, title=None,
                figsize=(10, 3), ylim=(None, None), alpha=0.4, ax=None,
                legend_loc='upper right'):

    # Take care of labels.
    if labels is None:
        labels = [None] * (len(ys) + 1)
    elif not isinstance(labels, list):
        labels = [labels]

    if not isinstance(xs, list):
        xs = [xs] * len(ys)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = None

    for x, (y_mean, y_std), label in zip(xs, ys, labels):
        ax.plot(x, y_mean, label=label)
        ax.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=alpha)
    ax.legend(loc=legend_loc)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(ylim)

    return fig, ax
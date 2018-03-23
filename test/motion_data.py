import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

from test_data import TestData


class MotionCaptureData(TestData):
    data_file = '../datasets/motion_data.pkl' # TODO make parameter
    motion_data = []

    """ 
    The dataset was taken from Reynolds, Zachs and Braver (2007) model of event segmentation.
    K = # of events to generate
    """
    def __init__(self, K=10):
        self.motion_data = pd.read_pickle(self.data_file)
        print "Number of events:", set(self.motion_data.EventNumber)
        print "Timepoint / event:", len(self.motion_data) / float(len(set(self.motion_data.EventNumber)))

        self.X, self.y = self.generate_random_events(K)
        self.X = scale(self.X)
        self.D = self.X.shape[1]

    def plot_event(self, event_number):
        fig, axes = plt.subplots(1, 2, figsize=(8.3, 4), gridspec_kw={'wspace': 0.3})
        e = self.motion_data.loc[self.motion_data.EventNumber == event_number, :]
        cols = e.columns[:-1]
        e = e[cols].values

        # loop through and plot each point as it moves in time.
        cc = sns.color_palette("YlOrRd", n_colors=e.shape[0])
        for ii in range(e.shape[0]-1):
            for jj in range(18):
                axes[0].plot([e[ii, jj], e[ii+1, jj]], [e[ii, 18+jj], e[ii+1, 18+jj]],
                         color=cc[ii], marker='^', markersize=ii)
        axes[0].set_xlabel('x-coordinate')
        axes[0].set_ylabel('y-coordinate')
        
        def connect(ax, t, color, linestyle, label=None):
            # plot the connections between the points in an ordered way
            kwargs = dict(color=color, linestyle=linestyle)
            ax.plot(e[t, 0:18], e[t, 18:36], 'o', color=color, label=label)
            ax.plot(e[t, 0:4], e[t, 18:22], **kwargs)
            ax.plot(e[t, 4:7], e[t, 22:25], **kwargs)
            ax.plot(e[t, [0,4]], e[t, [0,22]], **kwargs)
            ax.plot(e[t, [0,7]], e[t, [18, 18+7]], **kwargs)
            ax.plot(e[t, [7,8]], e[t, [18+7, 18+8]], **kwargs)
            ax.plot(e[t, [7,12]], e[t, [18+7, 18+12]], **kwargs)
            ax.plot(e[t, [8,12]], e[t, [18+8, 18+12]], **kwargs)
            ax.plot(e[t, 8:12], e[t, (18+8):(18+12)], **kwargs)
            ax.plot(e[t, 12:16], e[t, (18+12):(18+16)], **kwargs)
            ax.plot(e[t, [1,18]], e[t, [18+1, 18+18]], **kwargs)
            ax.plot(e[t, [4,18]], e[t, [18+4, 18+18]], **kwargs)
            ax.plot(e[t, [16,17]], e[t, [18+16, 18+17]], **kwargs) 
            ax.plot(e[t, [8,16]], e[t, [18+8, 18+16]], **kwargs)
            ax.plot(e[t, [12,16]], e[t, [18+12, 18+16]], **kwargs)
            
        connect(axes[1], 4, [0.5, 0.5, 0.5], '--', label='t=4')
        connect(axes[1], 5, 'k', '-', label='t=5')
        axes[1].legend()

    def generate_random_events(self, n_events):
        """

        Parameters
        ----------
        n_events: int
        """
        self.motion_data = pd.read_pickle(self.data_file)
        n_patterns = len(set(self.motion_data.EventNumber))

        X = []
        y = []
        for _ in range(n_events):
            p = np.random.randint(n_patterns)
            e = self.motion_data.loc[self.motion_data.EventNumber == p, :].values[:, :-1]
            X.append(e)
            y.append([p] * e.shape[0])
        return np.concatenate(X), np.concatenate(y)

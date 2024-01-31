import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_points_1d(x, y, outliers, x_test, y_test, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    ax.plot(x_test, y_test, c='b', label='true function')
    ax.scatter(x[~outliers], y[~outliers], marker='x', c='black', label='data')
    ax.scatter(x[outliers], y[outliers], marker='x', c='r', label='outliers')

    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('y', fontsize=14)

# Plot the results
def plot_gp_1d(x_test, post_mean, post_var, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    ci = 1.96 * np.sqrt(post_var)
    ax.plot(x_test, post_mean, label="GP mean", c='orange')
    ax.fill_between(x_test, (post_mean-ci), (post_mean+ci), color='green', alpha=.1, label="Confidence interval")

def plot_2d_3d(x, y, z, pos=111, fig=None):
    if fig is None:
        fig = plt.figure(figsize=(15, 5))
        pos = 111
    
    ax = fig.add_subplot(pos, projection='3d')
    ix_min = torch.argmin(z)
    ax.scatter(x.ravel()[ix_min], y.ravel()[ix_min], z[ix_min], c='deeppink', marker='*', s=500)
    ax.plot_surface(x, y, z.reshape(x.shape), cmap='coolwarm', alpha=.85)
    ax.set_xlabel('x')
    ax.set_xticks(np.arange(-5, 10.1, 5))
    ax.set_ylabel('y')
    ax.set_yticks(np.arange(0, 15.1, 5))
    ax.set_zlabel('z')

def plot_2d_2d(x, y, z, min=False, ax=None):
    if ax is None:
        ax = plt.gca()
    
    if min:
        ix_min = torch.argmin(z)
        ax.scatter(x.ravel()[ix_min], y.ravel()[ix_min], z[ix_min], c='deeppink', marker='*')
    ax.contourf(x, y, z.reshape(x.shape), cmap='coolwarm', alpha=.85)
    ax.set_xlabel('x')
    ax.set_xticks(np.arange(-5, 10.1, 5))
    ax.set_ylabel('y')
    ax.set_yticks(np.arange(0, 15.1, 5))

def plot_2d_points(x, y, outliers, ax=None):
    if ax is None:
        ax = plt.gca()
    
    ax.scatter(x[~outliers, 0], x[~outliers, 1], marker='x', c='black', label='data')
    ax.scatter(x[outliers, 0], x[outliers, 1], marker='x', c='r', label='outliers')
    ax.set_xlabel('x')
    ax.set_xticks(np.arange(-5, 10.1, 5))
    ax.set_ylabel('y')
    ax.set_yticks(np.arange(0, 15.1, 5))

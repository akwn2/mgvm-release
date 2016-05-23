"""
plotting.py
Plotting utility functions
"""
from matplotlib.pyplot import *
import matplotlib.cm as cm
import matplotlib.font_manager as fnt
import matplotlib.colors as colors
import numpy as np


# Redefine the defaults for imshow to be suitable for matrix and such
def matshow(x, cmap='rainbow'):
    """
    imshow with no interpolation and automatic aspect-ratio adjustment
    :param x: data
    :param cmap: color map
    :return:
    """
    imshow(x, interpolation='none', aspect='auto', cmap=cm.get_cmap(name=cmap))


def rose(axes, x, n_bins=10, color='#96e8ed', edgecolor='#2665f7'):
    x = np.mod(x, 2 * np.pi)
    hist, bins = np.histogram(x, bins=n_bins, range=(0, 2 * np.pi))

    bars = axes.bar(left=bins[0:-1], height=hist, width=bins[1] - bins[0], color=color, edgecolor=edgecolor)

    return hist, bins


def tides(th, p, y_t, y_v, pid_t, pid_v):

    train_ports = pid_t.flatten()
    valid_ports = pid_v.flatten()
    n_ports = 51

    fig = figure()
    ax_list = list()
    for ii in xrange(0, 6):
        for jj in xrange(0, 9):
            if 9 * ii + jj < 51:
                ax_list.append(fig.add_subplot(6, 9, jj + 9 * ii + 1, projection='polar'))

    for ii in xrange(0, n_ports):
        if ii in train_ports:
            hist, bins = rose(ax_list[ii], y_t[train_ports == ii], 20,
                              color='#c377e4', edgecolor='#8808bf')  # magenta
        else:
            hist, bins = rose(ax_list[ii], y_v[valid_ports == ii], 20,
                              color='#96e8ed', edgecolor='#2665f7')  # light blue

        scale_factor = np.max(hist) / np.max(p[:, ii])

        ax_list[ii].plot(th, p[:, ii] * scale_factor, color='#FF6D0D', linewidth=1.5)  # light orange
        ax_list[ii].set_title('Port:' + str(ii), fontsize=8)
        ax_list[ii].set_thetagrids([0, 90, 180, 270], frac=1.35)
        ax_list[ii].set_xticklabels(['0', '6', '12', '18'])
        ax_list[ii].set_yticklabels([])

        ticks_font = fnt.FontProperties(style='normal', size=6, weight='normal', stretch='normal')
        for label in ax_list[ii].get_xticklabels():
            label.set_fontproperties(ticks_font)

    fig.subplots_adjust(hspace=1.5)
    return fig


def circular_error_bars(th, p, norm=False):

    res = np.alen(th)

    tick1 = np.floor(0.25 * res)
    tick2 = np.floor(0.50 * res)
    tick3 = np.floor(0.75 * res)

    fig = figure(figsize=(3.50, 3.25), dpi=100)
    if norm:
        # bounds = np.linspace(-np.min(p), np.max(p), 50)
        # norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
        norm = colors.SymLogNorm(linthresh=0.1)
        imshow(p, aspect='auto', origin='lower', norm=norm, cmap=cm.get_cmap('Oranges'))
    else:
        imshow(p, aspect='auto', origin='lower', cmap=cm.get_cmap('Oranges'))

    yticks([0, tick1, tick2, tick3, res], ['$-\pi$', '$-\pi/2$', '$0$', '$+\pi/2$', '$+\pi$'])

    # Then scale the predicted and training sets to match the dimensions of the heatmap
    scaling_x = np.alen(p[0, :])
    scaling_y = res / (2 * np.pi)
    offset_y = tick2
    grid()

    # Fix axes
    axes = gca()
    axes.set_xlim(0, p.shape[1])

    return fig, scaling_x, scaling_y, offset_y
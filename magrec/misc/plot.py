import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import torch

from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib
from matplotlib import cm


def plot_n_components(
        data: list | tuple | np.ndarray | torch.Tensor,
        axes: list[plt.Axes] = None, units: str = '',
        norm: matplotlib.colors.Normalize = None,
        cmap: matplotlib.colors.Colormap | str | list | tuple = 'viridis',
        labels: list[str] | None = None,
        imshow_kwargs: dict | None = None,
        show: bool = False,
        symmetric: bool = True,
        climits: tuple = None,
        norm_type: str = 'row',
        alignment: str = 'horizontal',
) -> plt.Figure | list[plt.Figure]:
    """
    Plots n_components of a field in the provided axes or creates a new figure with axes.

    Args:
        axes (list):                        list of n_components axes
        data (torch.Tensor or numpy.array): field distribution to be plotted, shape (3, n_x, n_y), each component will be
                                            plotted on a separate axis or it is a list of 2d arrays of the same shape (one for each component)
        units (str):                        units of the field to display on the colorbar, e.g. 'mT' or 'A/m'
        norm (matplotlib.colors.Normalize): normalization object to share between different plot, for example
        cmap (str, matplotlib.colors.Colormap): colormap to use, default: viridis, 'cause it looks cool for magnetic fields
        labels (list[str]):                 list of labels to label components with, e.g. ['x', 'y', 'z'] etc
        imshow_kwargs (dict, None):         kwargs to pass to .imshow()
        show (bool):                        whether to keep the plot or .close() it, if show is True, the plot will be shown in notebooks
        symmetric (bool):                   whether to symmetrize the colorbar so that 0 is in the middle of the colormap and lower and upper data limits
                                            have the same absolute value
        norm_type ('row', 'all', str):      type of the normalization: per row ('row'), common to all ('all'), or by row with grouping,
                                            e.g. 'AAB' for 1st and 2nd rows to have the same norm, 3rd row to have its own norm
        alignment (str):                    alignment of the same quantity components, either 'horizontal' or 'vertical'
                                            if 'horizontal', then components will be plotted in the same row, if 'vertical', then in the same column

    Returns:
        fig (plt.Figure, list[plt.Figure]): figure or a list of figures with n_components
    """
    plt.rcParams['text.usetex'] = True

    if imshow_kwargs is None:
        imshow_kwargs = {}

    if isinstance(data, (torch.Tensor, np.ndarray)):
        shape = data.shape
        if len(shape) == 2:
            # we are dealing with a single component, add a singleton dimension
            data = data[None, ...]
            shape = data.shape
        
        n_components = shape[-3]
        # expect number of components to be in this dimension, if passed numpy.ndarray or torch.Tensor
        # handle the case when a list of rows is passed, then output each row in a new figure
        if len(shape) == 4:
            n_rows = shape[0]
        # otherwise think that there is only one set of components to plot and one row
        elif len(shape) == 3:
            n_rows = 1
            # add a dimension for the number of rows if it is not present
            data = data.unsqueeze(0) if isinstance(data, torch.Tensor) else data[np.newaxis, ...]
        else:
            raise ValueError(f'Expected data to have 3 or 4 dimensions, got {len(shape)}')
    elif isinstance(data, (list, tuple)):
        n_rows = len(data)
        # assume every row has the same number of components
        n_components = len(data[0])
        # when list of tuple is passed, it is ([nx, ny], [nx, ny],…) maps, or at least we cannot say better for now
        # it could happen that a list with shape length 4 is passed: n_maps, n_components, n_x, n_y, but we will not handle it here
    else:
        raise TypeError('Data must be a torch.Tensor, numpy.array, '
                        'list or tuple, got {}'.format(type(data)))

    if isinstance(data, torch.Tensor):
        # handle the case when input is torch.Tensor, since it may still require
        # autograd and be attached to the current computation graph, so we .detach() it
        data = data.detach().cpu().numpy()

    def get_color_map(cmap):
        if isinstance(cmap, str):
            cmap = cm.get_cmap(cmap)
        elif isinstance(cmap, matplotlib.colors.Colormap):
            pass
        else:
            raise TypeError('`cmap` must be a string with colormap name '
                            'or matplotlib.colors.Colormap, got {}'.format(type(cmap)))
        return cmap

    if labels is None:
        if n_components <= 6:
            labels = ['x', 'y', 'z', 'w', 'u', 'v'][:n_components]
        else:
            labels = [f'c{i}' for i in range(n_components)]

    if axes is not None:
        assert len(axes) == n_components * n_rows, \
            "There should be enough axes to plot {} components, " \
            "got {}".format(n_components * n_rows, len(axes))
        fig = axes[0].get_figure()
        # Assume that axes already have the associated color axis, as in case when it was created by ImageGrid
        cax = axes[0].cax
    else:
        fig = plt.figure()
        if alignment == "horizontal":
            grid = ImageGrid(fig, rect=(0, 0, 1, 1),
                            nrows_ncols=(n_rows, n_components),
                            axes_pad=0.05,
                            label_mode="1",
                            share_all=True,
                            cbar_location="right",
                            cbar_mode="edge",
                            cbar_size="10%",
                            cbar_pad=0.05,
                            )
        elif alignment == "vertical":
            grid = ImageGrid(fig, rect=(0, 0, 1, 1),
                            nrows_ncols=(n_components, n_rows),
                            axes_pad=0.05,
                            label_mode="1",
                            share_all=True,
                            cbar_location="bottom",
                            cbar_mode="edge",
                            cbar_size="10%",
                            cbar_pad=0.3,
                            )
        axes = grid.axes_all

    # use default colormap if not specified
    c = get_color_map('viridis') if cmap is None else cmap

    if norm_type == 'row':
        norm_type = 'ABCDEFFGHIJKLMNOPQRSTUVWXYZ'[:n_rows]

    if norm_type == 'all':
        norm_type = 'A' * n_rows

    norm_dict = {key: None for key in norm_type}
    for i, norm_group in enumerate(norm_type):
        row_data = data[i]
        if norm_dict[norm_group] is None and climits is None:
            norm_dict[norm_group] = get_color_norm(row_data, symmetric=symmetric)
        elif climits is not None:
            norm_dict[norm_group] = get_color_norm(vmin=min(climits), vmax= max(climits), symmetric=False)
        else:
            new_norm = get_color_norm(row_data, symmetric=symmetric)
            norm_dict[norm_group].vmin = min(norm_dict[norm_group].vmin, new_norm.vmin)
            norm_dict[norm_group].vmax = max(norm_dict[norm_group].vmax, new_norm.vmax)

    for i, ax in enumerate(axes):
        ax: plt.Axes
        ax.set_aspect(aspect='equal', adjustable='box', anchor='NW')

        # take // to get integer division to understand which row, and take % to get which component (column)
        # REMARK: Maybe I should report the problem with imshow that when the data is exactly 0, and the limits are smaller than 1e-9,
        # and the dtype of the data is float32 or smaller, then imshow shows wrong colors, as explained below:
        datum = np.float64(data[i // n_components][i % n_components]).T  # convert to float64 to avoid an issue with matplotlib and casting
        # the problems arises when vmin and vmax is small (< 1e-9) and the data is very close to or is exactly 0
        # in those cases the color is like it is 0 on the colormap, even though 0 should be mapped by norm to 0.5

        # # do once at the very beginning for all axes
        # if i == 0:
        #     if norm_type == 'all':
        #         norm = get_color_norm(datum, symmetric=symmetric) if norm is None else norm

        # do at the beginning of row
        if i % n_components == 0:
            # assign colormap to the row if colormap is given as a list | tuple (cmap1, cmap2, ...)
            if isinstance(cmap, (list, tuple)):
                c = cmap[i // n_components]
                if c is Ellipsis and i == 0:
                    c = get_color_map('viridis')
                elif c is Ellipsis:
                    # if Ellipsis is passed, use the same colormap as for the previous row
                    c = get_color_map(cmap[i // n_components - 1])
                elif isinstance(c, (str, matplotlib.colors.Colormap)):
                    c = get_color_map(c)

        norm = norm_dict[norm_type[i // n_components]]

        im = ax.imshow(datum, cmap=c, norm=norm, origin='lower',
                       **imshow_kwargs)  # extent=(0, 1, 0, 1))
        # https://github.com/matplotlib/matplotlib/issues/16910 — seems to be related
        ax.set_label(labels[i % n_components])
        if cmap == 'bwr':
            label_color = 'black'
        else:
            label_color = None
        add_inner_title(ax, labels[i % n_components], loc='upper left', color=label_color)

        if (i + 1) % n_components == 0:
            # if last component, plot the colorbar
            cbar = ax.cax.colorbar(im)
            cbar.set_label(units)

    # Make an array of tick positions from 0 to datum.shape[0] so that there are at least 5 ticks
    # and the last tick is at the end of the array
    xticks = np.arange(0, datum.shape[1], max(20, datum.shape[1] // 5))
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    
    yticks = np.arange(0, datum.shape[0], max(20, datum.shape[0] // 5))
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks)

    # On the last axis, show an inset with coordinate system directions: arrows pointing to the right and up
    ax_inset = inset_axes(ax, width="30%", height="30%", loc='lower left')
    ax_inset.set_aspect('equal')
    ax_inset.set_axis_off()
    ax_inset.arrow(0, 0, 1, 0, head_width=0.3, head_length=0.3, linewidth=0.3, capstyle='butt', facecolor='k', edgecolor='k')
    ax_inset.arrow(0, 0, 0, 1, head_width=0.3, head_length=0.3, linewidth=0.3, capstyle='butt', facecolor='k', edgecolor='k')
    ax_inset.text(1.5, 0, r'$x$', fontsize=12, color='k')
    ax_inset.text(0, 1.7, r'$y$', fontsize=12, color='k')


    fig.subplots_adjust(hspace=None)

    if not show:
        plt.close()

    return fig


def get_color_norm(z=None, vmin=None, vmax=None,
                   symmetric=False) -> matplotlib.colors.Normalize:
    if z is None:
        if vmin is None and vmax is None:
            ValueError(
                'Need to work with something to get the norm, instead vmax and xmin are None and z is None!')
        elif not (vmin and vmax):
            ValueError(
                f'Both vmin and vmax need to be provided, but {vmax=} and {vmin=}')

    vmin = vmin if vmin is not None else z.min()
    vmax = vmax if vmax is not None else z.max()
    if symmetric:
        # make 0 in the middle by specifying same amount of values above and below it
        bound = np.max(np.abs([vmin, vmax]))
        vmin = -bound
        vmax = bound

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    return norm


def add_inner_title(ax: plt.Axes, title: str, loc, color=None, **kwargs):
    from matplotlib.offsetbox import AnchoredText
    properties = {'size': plt.rcParams['legend.fontsize']}
    # add an option to adjust color
    if color is not None:
        properties['color'] = color
    # got it from here: https://matplotlib.org/stable/gallery/axes_grid1/demo_axes_grid2.html
    at = AnchoredText(title, loc=loc, prop=properties, pad=0., borderpad=0.5,
                      frameon=False, **kwargs)
    # add text to the axis
    ax.add_artist(at)
    return at

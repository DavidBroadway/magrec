import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from PIL import Image


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
        show_coordinate_system: bool = True,
        zoom_in_region=None,
        title: str = '',
) -> plt.Figure | list[plt.Figure]:
    """
    Plots n_components of a field in the provided axes or creates a new figure with axes.

    Args:
        axes (list):                        list of n_components axes
        data (torch.Tensor or numpy.array): field distribution to be plotted, shape (n_components, n_x, n_y), each component will be
                                            plotted on a separate axis or it is a list of 2d arrays of the same shape (one for each component)
        units (str):                        units of the field to display on the colorbar, e.g. 'mT' or 'A/m'
        norm (matplotlib.colors.Normalize): normalization object to share between different plot, for example
        cmap (str, matplotlib.colors.Colormap): colormap to use, default: viridis, 'cause it looks cool for magnetic fields
        labels (list[str] or str):          list of labels to label components with, e.g. ['x', 'y', 'z'] etc, if str 'no_labels' is passed then no labels will be shown
        imshow_kwargs (dict, None):         kwargs to pass to .imshow()
        show (bool):                        whether to keep the plot or .close() it, if show is True, the plot will be shown in notebooks
        symmetric (bool):                   whether to symmetrize the colorbar so that 0 is in the middle of the colormap and lower and upper data limits
                                            have the same absolute value
        norm_type ('row', 'all', str):      type of the normalization: per row ('row'), common to all ('all'), or by row with grouping,
                                            e.g. 'AAB' for 1st and 2nd rows to have the same norm, 3rd row to have its own norm
                                            If alignment is 'vertical' treat rows as columns and vice versa. Default: 'row'
        alignment (str):                    alignment of the same quantity components, either 'horizontal' or 'vertical'. 
                                            if 'horizontal', then components will be plotted in the same row, if 'vertical', then in the same column
        show_coordinate_system (bool):      whether to show the coordinate system in the plot, default: True
        zoom_in_region (bool):              whether to show an inset with a zoomed-in region of the plot, default: None
                                            To specify the region, pass a list of two tuples, where the tuples contain the coordinates of the 
                                            bottom-left (BL) and top-right (TR) and corners of the rectangle: [(BL_x, BL_y), (TR_x, TR_y)]. 
                                            Shows the inset only for the first drawn component for now.
        title (list(str) or str):           if str, then it is the title of the plot, if list, then it is a list of titles for each row

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
        if isinstance(data[0], (list, tuple)):
            n_components = len(data[0])
        else:
            # deal with the case when components are passed as a list of tensors
            n_components = len(data)
            n_rows = 1
            data = torch.stack(data, dim=0).unsqueeze(0)
            
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
    
    if isinstance(labels, str):
        if labels == 'no_labels':
            labels = [''] * n_components
        else:
            raise ValueError('If `labels` is a string, it must be "no_labels", got {}'.format(labels))

    ## 
    # Create axes or create a figure here
    if isinstance(axes, list):
        assert len(axes) == n_components * n_rows, \
            "There should be enough axes to plot {} components, " \
            "got {}".format(n_components * n_rows, len(axes))
        fig = axes[0].get_figure()
        # Assume that axes already have the associated color axis, as in case when it was created by ImageGrid
        cax = axes[0].cax
    elif isinstance(axes, plt.Axes):
        axes = [axes]
        fig = axes[0].get_figure()
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
                            direction="row",
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
                            direction="column",
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
            
    # Handle the inset: optionally display a zoomed-in region of the original plot
    if zoom_in_region is not None:
        x0, y0 = zoom_in_region[0]
        x1, y1 = zoom_in_region[1]
        axins = inset_axes(ax, width="30%", height="30%", loc='upper left')
        inset_datum = datum[x0:x1, y0:y1]
        axins.imshow(inset_datum, cmap=cmap, origin='lower')
        
        axins.set_xticks([])
        axins.set_yticks([])

    # Make an array of tick positions from 0 to datum.shape[0] so that there are at least 5 ticks
    # and the last tick is at the end of the array
    xticks = np.arange(0, datum.shape[1], max(20, datum.shape[1] // 5))
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    
    yticks = np.arange(0, datum.shape[0], max(20, datum.shape[0] // 5))
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks)

    # On the last axis, show an inset with coordinate system directions: arrows pointing to the right and up
    if show_coordinate_system:
        ax_inset = inset_axes(ax, width="30%", height="30%", loc='lower left')
        ax_inset.set_aspect('equal')
        ax_inset.set_axis_off()
        ax_inset.arrow(0, 0, 1, 0, head_width=0.3, head_length=0.3, linewidth=0.3, capstyle='butt', facecolor='k', edgecolor='k')
        ax_inset.arrow(0, 0, 0, 1, head_width=0.3, head_length=0.3, linewidth=0.3, capstyle='butt', facecolor='k', edgecolor='k')
        ax_inset.text(1.5, 0, r'$x$', fontsize=12, color='k')
        ax_inset.text(0, 1.7, r'$y$', fontsize=12, color='k')

    fig.subplots_adjust(hspace=None)
    
    if title:
        if isinstance(title, str):
            # Use .annotate() method to add the title text at the top of the figure
            axes[0].annotate(title, xy=(0.5, 1.1), xycoords='axes fraction',
                             xytext=(0, 0), textcoords='offset points',
                             ha='center', va='baseline')
        elif isinstance(title, (list)):
            raise NotImplementedError('Not implemented yet')

    if not show:
        plt.close()

    return fig


def plot_vector_field_2d(current_distribution, ax: plt.Axes = None, interpolation='none', cmap='plasma',
                         show=False, num_arrows=20, zoom_in_region=None):
    """
    Visualizes the current distribution in 2D as a heatmap of magnitudes and arrows representing the flow direction.
    Additionally, the function can render an inset region from the given current distribution with an arrow indicating 
    the average flow direction in that region.
    
    Parameters:
    ----------
    current_distribution : ndarray
        An array of shape (2, W, H) representing current distribution vectors.
    
    interpolation : str, optional
        The interpolation method to be used for the heatmap. Options are as provided by matplotlib's `imshow`.
        Default is 'none'.
    
    cmap : str, optional
        The colormap to be used for the heatmap. Options are as provided by matplotlib's colormaps.
        Default is 'plasma'.
    
    show : bool, optional
        If True, the plot is displayed using `plt.show()`. Otherwise, the plot is returned without being shown.
        Default is False.
    
    num_arrows : int, optional
        The approximate number of arrows to be used for the quiver plot across the width of the image.
        The function will adjust the number of arrows to avoid overcrowding. Default is 20.
    
    zoom_in_region : tuple of two tuples, optional
        A tuple representing the top-left and bottom-right corners of the rectangle for the inset. The format is 
        ((index_x_origin, index_y_origin), (index_x_diagonal, index_y_diagonal)). If provided, the function will render 
        the specified inset region with an arrow indicating the average flow direction. Default is None.
    
    Returns:
    -------
    fig : Figure
        A matplotlib figure containing the visualizations.

    Notes:
    -----
    - The main visualization provides a heatmap of the magnitudes and arrows indicating the flow direction of the 
      current distribution.
    - The inset, if specified, provides a zoomed-in view of a particular region of the main visualization, allowing for 
      a closer inspection of details.

    Examples:
    --------
    >>> dist = np.random.rand(2, 100, 100)
    >>> fig = plot_vector_field_2d(dist, interpolation='bilinear', cmap='inferno', show=True, num_arrows=30, inset_region=((20, 20), (40, 40)))
    """  
    # PARAMS
    figsize = (8, 8)
    
    # Create a figure with an axis if none is provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad="4%")
    else:
        fig = ax.get_figure()
        
    if isinstance(current_distribution, torch.Tensor):
        current_distribution = current_distribution.detach().cpu().numpy()
        
    W, H = current_distribution.shape[-2:]
    step_size = max(W, H) // num_arrows
    
    # Create averaged grid
    x_centers = np.arange(step_size // 2, W, step_size)
    y_centers = np.arange(step_size // 2, H, step_size)
    
    x_grid, y_grid = np.meshgrid(x_centers, y_centers, indexing='ij')
    
    # Initialize averaged fields
    avg_u = np.zeros((len(x_centers), len(y_centers)))
    avg_v = np.zeros((len(x_centers), len(y_centers)))
    avg_m = np.zeros((len(x_centers), len(y_centers)))
    
    # Loop through and average
    for i, x in enumerate(x_centers):
        for j, y in enumerate(y_centers):
            x_low, x_high = x - step_size // 2, x + step_size // 2 + 1
            y_low, y_high = y - step_size // 2, y + step_size // 2 + 1
            
            avg_u[i, j] = np.mean(current_distribution[0, x_low:x_high, y_low:y_high])
            avg_v[i, j] = np.mean(current_distribution[1, x_low:x_high, y_low:y_high])
            avg_m[i, j] = np.sqrt(avg_v[i, j] ** 2 + avg_u[i, j] ** 2)
    
    magnitudes = np.hypot(current_distribution[0], current_distribution[1]).transpose(1, 0)
    
    im = ax.imshow(magnitudes, interpolation=interpolation, cmap=cmap, origin='lower')
    
    fig.colorbar(im, cax=cax, orientation='vertical', label='Magnitude')
    
    # Compute scale of the arrow length. 
    # Scale gives number of data points per arrow length unit, 
    # e.g. A/mm^2 per plot width. We want the maximum arrow length to be 1/num_arrows of the plot width.
    # How much data units per arrow length unit? 
    scale = 0.95 * avg_m.max() * num_arrows  # 0.95 to make the length of the longest arrow a bit longer 
                                             # than the spacing between arrows blocks.  
    
    ax.quiver(
        x_grid, y_grid, avg_u, avg_v, color='black',
        pivot='mid', units='width', angles='uv',
        scale=scale, scale_units='width',
    )
    
    if zoom_in_region is not None:
        x0, y0 = zoom_in_region[0]
        x1, y1 = zoom_in_region[1]
        axins = inset_axes(ax, width="30%", height="30%", loc='upper left')
        axins.imshow(magnitudes[x0:x1, y0:y1], cmap=cmap, origin='lower')
        
        axins.set_xticks([])
        axins.set_yticks([])
        
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal')
        
    return fig

def plot_to_tensorboard(writer, fig, tag, step):
    """
    Takes a matplotlib figure handle and converts it using
    canvas and string-casts to a numpy array that can be
    visualized in TensorBoard using the add_image function

    Parameters:
        writer (tensorboard.SummaryWriter): TensorBoard SummaryWriter instance.
        fig (matplotlib.pyplot.fig): Matplotlib figure handle.
        tag (str): Name for the figure in TensorBoard.
        step (int): counter usually specifying steps/epochs/time.
    """

    # Draw figure on canvas
    matplotlib.use('Agg', force=True)
    fig.canvas.draw_idle()  # Updates the canvas
    fig.canvas.draw()
    
    renderer = fig.canvas.renderer
    raw_data = renderer.tostring_rgb()
    size = int(renderer.width), int(renderer.height)

    # Convert the figure to numpy array, read the pixel values and reshape the array
    pil_image = Image.frombytes('RGB', size, raw_data)
    img = np.array(pil_image)

    # Normalize into 0-1 range for TensorBoard(X). Swap axes for newer versions where API expects colors in first dim
    # img = img / 255.0
    img = np.swapaxes(img, 0, 2) # if your TensorFlow + TensorBoard version are >= 1.8
    img = np.swapaxes(img, 1, 2) # if your TensorFlow + TensorBoard version are >= 1.8

    # Add figure in numpy "image" to TensorBoard writer
    writer.add_image(tag, img, step)
    plt.close(fig)
    

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


def set_backend(backend):
    matplotlib.use(backend=backend, force=True)
    
def ion():
    plt.ion()
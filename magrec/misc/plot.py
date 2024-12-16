import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from matplotlib import cm
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from PIL import Image

import vedo

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "Helvetica"
})

      
# PARAMS shared across plotting function. Allows consistant size of plots.   
PAD_INCH = 0.5  # pad size around plots, in inches
SIZE_INCH = 2   # base size of the plots, in inches, the base size is used to calculate width and height of 
# individual plots using the ratio of number of pixels in x and y directions
CBAR_PORTION = 0.05  # portion of the single plot size to give to colorbar
        

# TODO: add scaling factor, to be able to plot 2× component
# TODO: add choice of label color, or heuristic to choose it automatically based on the colormap and background color
def plot_n_components(
        data: list | tuple | np.ndarray | torch.Tensor,
        axes: list[plt.Axes] = None, 
        units: str = '',
        norm: matplotlib.colors.Normalize = None,
        cmap: matplotlib.colors.Colormap | str | list | tuple = 'viridis',
        labels: list[str] | None = None,
        imshow_kwargs: dict | None = None,
        show: bool = False,
        symmetric: bool = True,
        climits: tuple = None,
        norm_type: str = 'map',
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
                                            Implicit limit of maximum 27 maps to be plotted. The secret reason is in norm_type handling. 
        units (str):                        units of the field to display on the colorbar, e.g. 'mT' or 'A/m'
        norm (matplotlib.colors.Normalize): normalization object to share between different plot, for example
        cmap (str, matplotlib.colors.Colormap): colormap to use, default: viridis, 'cause it looks cool for magnetic fields, can also be a list of colormaps or their names 
        labels (list[str] or str):          list of labels to label components with, e.g. ['x', 'y', 'z'] etc, if str 'no_labels' is passed then no labels will be shown
        imshow_kwargs (dict, None):         kwargs to pass to .imshow()
        show (bool):                        whether to keep the plot or .close() it, if show is True, the plot will be shown in notebooks
        symmetric (bool):                   whether to symmetrize the colorbar so that 0 is in the middle of the colormap and lower and upper data limits
                                            have the same absolute value
        norm_type ('map', 'all', str):      type of the normalization: per map ('map'), common to all ('all'), or by row/column with grouping,
                                            e.g. 'AAB' for 1st and 2nd maps to have the same norm, 3rd map to have its own norm. Default: 'map'
                                            Allows to have common normalization for different maps, e.g. when they both show similar fields to compare
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
            n_maps = shape[0]
        # otherwise think that there is only one set of components to plot and one row
        elif len(shape) == 3:
            n_maps = 1
            # add a dimension for the number of rows if it is not present
            data = data.unsqueeze(0) if isinstance(data, torch.Tensor) else data[np.newaxis, ...]
        else:
            raise ValueError(f'Expected data to have 3 or 4 dimensions, got {len(shape)}')
    elif isinstance(data, (list, tuple)):
        n_maps = len(data)
        # assume every row has the same number of components
        if isinstance(data[0], (list, tuple)):
            n_components = len(data[0])
        elif isinstance(data[0], (torch.Tensor, np.ndarray)):
            n_components = data[0].shape[0]
        else:
            # deal with the case when components are passed as a list of tensors
            n_components = len(data)
            n_maps = 1
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
        if labels == 'no_labels' or labels == '':
            labels = [''] * n_components
        else:
            raise ValueError('If `labels` is a string, it must be "no_labels" or an empty string \"\", got {}'.format(labels))

    ## 
    # Create axes or create a figure here
    if isinstance(axes, list):
        assert len(axes) == n_components * n_maps, \
            "There should be enough axes to plot {} components, " \
            "got {}".format(n_components * n_maps, len(axes))
        fig = axes[0].get_figure()
        # Assume that axes already have the associated color axis, as in case when it was created by ImageGrid
        cax = axes[0].cax
    elif isinstance(axes, plt.Axes):
        axes = [axes]
        fig = axes[0].get_figure()
        cax = axes[0].cax
    else:
        # We need to approximately calculate the size of the figure according to the number of components and fields, and also the size of each
        # component. That is needed to match size of the colorbar, that is otherwise larger than the component maps instead.
        figsize = get_figsize(n_components, n_maps, alignment)
        
        # Create the figure according to the calculated size
        fig = plt.figure(figsize=figsize)  
        gs = get_gridspec(n_components=n_components, n_maps=n_maps, alignment=alignment)
        # Note to self: gs indexing is gs[row_index, column_index].

    # use default colormap if not specified
    c = get_color_map('viridis') if cmap is None else cmap

    if norm_type == 'map':
        norm_type = 'ABCDEFFGHIJKLMNOPQRSTUVWXYZ'[:n_maps]
    elif norm_type == 'all':
        norm_type = 'A' * n_maps
    else:
        if len(norm_type) != n_maps:
            raise ValueError('`norm_type` must be a string of length equal to the number of maps, '
                            'got {} with length {} instead of required length {}'.format(norm_type, len(norm_type), n_maps))

    # Handle case when normalization is different for different groups of components
    norm_dict = {key: None for key in norm_type}
    for i, norm_group in enumerate(norm_type):
        row_data = data[i]
        if norm_dict[norm_group] is None and climits is None:
            norm_dict[norm_group] = get_color_norm(row_data, symmetric=symmetric)
        elif climits is not None:
            norm_dict[norm_group] = get_color_norm(vmin=min(climits), vmax=max(climits), symmetric=False)
        else:
            new_norm = get_color_norm(row_data, symmetric=symmetric)
            norm_dict[norm_group].vmin = min(norm_dict[norm_group].vmin, new_norm.vmin)
            norm_dict[norm_group].vmax = max(norm_dict[norm_group].vmax, new_norm.vmax)

    # for i, ax in enumerate(axes):
    for map_n in range(n_maps):
        for component_n in range(n_components):
            if alignment == "horizontal":
                subplot_spec = gs[map_n, component_n]
            elif alignment == "vertical":
                subplot_spec = gs[1 + component_n, map_n]
            
            if axes is None:
                ax = fig.add_subplot(subplot_spec)            
            else:
                ax = axes[map_n * n_components + component_n]
                
            # Add title to the plot per component group, if title is a list of strings
            if isinstance(title, list) and component_n == 0:
                ax.set_title(title[map_n], fontsize=10)
                # Increase the space between rows if there are titles
                if alignment == "horizontal":
                    gs.update(hspace=PAD_INCH * 1.1)
                elif alignment == "vertical":
                    gs.update(wspace=PAD_INCH * 1.1)
            
            ax: plt.Axes                 
            ax.set_aspect(aspect='equal', adjustable='box', anchor='NW')

            # REMARK: Maybe I should report the problem with imshow that when the data is exactly 0, and the limits are smaller than 1e-9,
            # and the dtype of the data is float32 or smaller, then imshow shows wrong colors, as explained below:
            datum = np.float64(data[map_n][component_n]).T  # convert to float64 to avoid an issue with matplotlib and casting
            # the problems arises when vmin and vmax is small (< 1e-9) and the data is very close to or is exactly 0
            # in those cases the color is like it is 0 on the colormap, even though 0 should be mapped by norm to 0.5

            norm = norm_dict[norm_type[map_n]]

            im = ax.imshow(datum, cmap=c, norm=norm, origin='lower',
                        **imshow_kwargs)  # extent=(0, 1, 0, 1))
            # https://github.com/matplotlib/matplotlib/issues/16910 — seems to be related
            ax.set_label(labels[component_n])
            
            if cmap == 'bwr':
                label_color = 'black'
            else:
                label_color = None
            
            add_inner_title(ax, labels[component_n], loc='upper left', color=label_color)
            
            if subplot_spec.is_first_col() and subplot_spec.is_last_row():
                # Make an array of tick positions from 0 to datum.shape[0] so that there are at least 5 ticks
                # and the last tick is at the end of the array
                xticks = np.arange(0, datum.shape[1], max(20, datum.shape[1] // 5))
                ax.set_xticks(xticks)
                ax.set_xticklabels(xticks)
                
                yticks = np.arange(0, datum.shape[0], max(20, datum.shape[0] // 5))
                ax.set_yticks(yticks)
                ax.set_yticklabels(yticks)     
            else: 
                ax.set_xticks([])
                ax.set_yticks([])
                
        # Using datum, rescale figsize to match the aspect ratio of the data
        x2y_ratio = datum.shape[1] / datum.shape[0]
        # Calculate the width of the figure based on the height
        figsize = get_figsize(n_components, n_maps, alignment, x2y_ratio)
        fig.set_size_inches(figsize)
            
        # After going through all components, add a colorbar:    
        # get colormap for the map (collection of components) if colormap is given as a list | tuple (cmap1, cmap2, ...)
        if isinstance(cmap, (list, tuple)):
            c = cmap[map_n]
            if c is Ellipsis and map_n == 0:
                c = get_color_map('viridis')
            elif c is Ellipsis:
                # if Ellipsis is passed, use the same colormap as for the previous row
                c = get_color_map(cmap[map_n - 1])
        elif isinstance(c, (str, matplotlib.colors.Colormap)):
            c = get_color_map(c)
            
        if units:
            if '\mu' not in units:
                units = units.replace('u', '$\mu$')
        
        if alignment == "horizontal":
            # effectively pad the colorbar by 0.1 on each side
            cax_gs = gs[map_n, component_n + 1].subgridspec(nrows=3, ncols=1, height_ratios=[0.1, 1, 0.2], wspace=0, hspace=0)[1]
            cax = fig.add_subplot(cax_gs)
            cbar = fig.colorbar(im, cax=cax, orientation='vertical')
            if units: 
                cax.set_title(units, fontsize=9, loc='left')
        elif alignment == "vertical":
            # effectively pad the colorbar by 0.1 on each side
            cax_gs = gs[0, map_n].subgridspec(nrows=1, ncols=3, width_ratios=[0.1, 1, 0.2], wspace=0, hspace=0)[1]
            cax = fig.add_subplot(cax_gs)
            cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
            # locate ticks at the top of the colorbar
            cax.xaxis.tick_top()
            if units:
                cax.set_ylabel(units, fontsize=9, rotation='horizontal', x=0, y=3.7)
                # put label to the right of the colorbar
                cax.yaxis.set_label_position('right')
        
        # If units have "u" prefix in it, replace it with micro symbol
        # cbar.set_label(units)

            
    # Handle the inset: optionally display a zoomed-in region of the original plot
    if zoom_in_region is not None:
        x0, y0 = zoom_in_region[0]
        x1, y1 = zoom_in_region[1]
        axins = inset_axes(ax, width="30%", height="30%", loc='upper left')
        inset_datum = datum[x0:x1, y0:y1]
        axins.imshow(inset_datum, cmap=cmap, origin='lower')
        
        axins.set_xticks([])
        axins.set_yticks([])

    # On the last axis, show an inset with coordinate system directions: arrows pointing to the right and up
    if show_coordinate_system:
        ax_inset = inset_axes(ax, width="30%", height="30%", loc='lower left')
        ax_inset.set_aspect('equal')
        ax_inset.set_axis_off()
        ax_inset.arrow(0, 0, 1, 0, head_width=0.3, head_length=0.3, linewidth=0.3, capstyle='butt', facecolor='k', edgecolor='k')
        ax_inset.arrow(0, 0, 0, 1, head_width=0.3, head_length=0.3, linewidth=0.3, capstyle='butt', facecolor='k', edgecolor='k')
        ax_inset.text(1.5, 0, r'$x$', fontsize=12, color='k')
        ax_inset.text(0, 1.7, r'$y$', fontsize=12, color='k')
    
    if title:
        if isinstance(title, str):
            # Use .annotate() method to add the title text at the top of the figure
            plt.suptitle(title)

    if not show:
        plt.close()

    return fig


def plot_vector_field_2d(current_distribution, ax: plt.Axes = None, 
                         interpolation='none', cmap='plasma', units=None,
                         title=None, show=False, num_arrows=20, zoom_in_region=None):
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
            
    units : str, optional
        Show units on the colorbar. Default is None (no units).
        
    title: str, optional
        Title of the plot. Default is None (no title).
    
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
    # figsize = get_figsize(n_components=1, n_maps=1, alignment='horizontal')
    
    # Create a figure with an axis if none is provided
    if ax is None:
        fig = plt.figure(frameon=False)
        gs = get_gridspec(n_components=1, n_maps=1, alignment='horizontal')
        ax = fig.add_subplot(gs[0, 0])
        cax_gs = gs[0, 1].subgridspec(nrows=3, ncols=1, height_ratios=[0.1, 1, 0.2], wspace=0, hspace=0)[1]
        cax = fig.add_subplot(cax_gs)
    else:
        fig = ax.get_figure()
        try:
            cax = ax.cax
        except AttributeError:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
        
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
    
    im = ax.imshow(magnitudes, interpolation=interpolation, vmin=0, cmap=cmap, origin='lower')
    
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    if units:
        cax.set_title(units, fontsize=9, loc='left')
        
    if title:
        if isinstance(title, str):
            # Use .annotate() method to add the title text at the top of the figure
            ax.annotate(title, xy=(0.5, 1.1), xycoords='axes fraction',
                        xytext=(0, 0), textcoords='offset points',
                        ha='center', va='baseline')
    
    # Compute scale of the arrow length. 
    # Scale gives number of data points per arrow length unit, 
    # e.g. A/mm^2 per plot width. We want the maximum arrow length to be 1/num_arrows of the plot width.
    # How much data units per arrow length unit? 
    scale = 1.1 * avg_m.max() * num_arrows  # 1.1 to make the length of the longest arrow a bit shorter 
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
    
    if not show:
        plt.close()
        
    return fig
    

def get_color_norm(vals=None, vmin=None, vmax=None,
                   symmetric=False) -> matplotlib.colors.Normalize:
    """Returns a normalization object for the colorbar."""
    if vals is None:
        if vmin is None and vmax is None:
            ValueError(
                'Need to work with something to get the norm, instead vmax and xmin are None and z is None!')
        elif not (vmin and vmax):
            ValueError(
                f'Both vmin and vmax need to be provided, but {vmax=} and {vmin=}')

    vmin = vmin if vmin is not None else vals.min()
    vmax = vmax if vmax is not None else vals.max()
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

def plot_check_aligned_data(data_pts, data_vals):
    """Plot `data_pts` and `data_vals` so that to check whether pts coordinates
    correspond to correct vals, visually."""
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Plot each component of data_vals
    for i in range(3):
        vals = data_vals[:, i]
        norm = get_color_norm(vals, symmetric=True)
        sc = axs[i].scatter(data_pts[:, 0], data_pts[:, 1], c=vals, cmap='RdBu_r', norm=norm)
        axs[i].set_title(f'Component {i+1}')
        axs[i].set_xlabel('X')
        axs[i].set_ylabel('Y')
        fig.colorbar(sc, ax=axs[i])

    plt.show()

def get_figsize(n_components, n_maps, alignment, x2y_ratio=1):
    """Calculate the size of the figure based on the number of components and maps to be plotted."""
    
    if alignment == 'horizontal':
        ratio = x2y_ratio
    elif alignment == 'vertical':  
        ratio = 1 / x2y_ratio
    
    big = PAD_INCH + (SIZE_INCH * ratio + PAD_INCH) * n_components + CBAR_PORTION * SIZE_INCH + PAD_INCH
    short = PAD_INCH + SIZE_INCH / ratio * n_maps + PAD_INCH
    if alignment == 'horizontal':
        return (big, short)
    elif alignment == 'vertical':
        return (short, big)
    
    
def get_gridspec(n_components, n_maps, alignment):
    """Create matplotlib.gridspec.GridSpec object for the given number of components and maps.
    Components and maps are to be plotted in rows/columns, depending on the alignment. Space for colorbar is added,
    and is returned as a +1 column/row in the gridspec for each map.
    
    Spacing between plots is set proportional to colorbar size for visual balance.
    """
    # Make spacing proportional to colorbar size
    spacing = CBAR_PORTION * 0.8  # Slightly smaller than colorbar for balance
    
    if alignment == "horizontal":
        gs = gridspec.GridSpec(
            nrows=n_maps, 
            ncols=n_components + 1, 
            width_ratios=[1] * n_components + [CBAR_PORTION],
            wspace=spacing,
            hspace=spacing
        )
    elif alignment == "vertical":
        gs = gridspec.GridSpec(
            ncols=n_maps, 
            nrows=n_components + 1, 
            height_ratios=[CBAR_PORTION] + [1] * n_components,
            hspace=spacing,
            wspace=spacing
        )
    return gs



def plot_ffs_params(model, ax=None, in_3d=False):
    """Plot Fourier features parameters using matplotlib (2D) or vedo (3D).
    
    Args:
        model: Model containing Fourier features
        ax: matplotlib axes (for 2D only)
        in_3d: Force 3D visualization
        backend: 'mpl' or 'vedo'
    """
    # Get vectors from model
    try:
        vectors = model.B.detach().T
    except AttributeError:
        vectors = model.get_ffs_as_vectors()  # Shape (N, dim)
        
    dim = vectors.shape[1]
    
    # Use vedo for 3D or when forced
    if dim == 3 and in_3d:
        # Initialize vedo plot
        plotter = vedo.Plotter()
        
        axis_extent = np.abs(vectors).max() * 1.1 
        
        # Heuristics regarding the ball radius, to make it look proportional 
        # to the viewport size
        ball_radius = axis_extent / 20
        
        # Add balls at the vector end points with tooltips
        for i, v in enumerate(vectors):
            vector_norm = np.linalg.norm(v)
            tooltip_text = f"Coords: {v}\nNorm: {vector_norm:.3f}"
            
            # Add a ball (sphere) at the vector's end point
            ball = vedo.Sphere(pos=v, r=np.float32(ball_radius), c='red', alpha=0.7)
            ball.name = f"Vector {i}"
            plotter += ball            
            
        # Add central axes
        axes = vedo.Axes(
            xrange=(-axis_extent,axis_extent),
            yrange=(-axis_extent,axis_extent),
            zrange=(-axis_extent,axis_extent),
            xtitle='x',
            ytitle='y',
            ztitle='z',
            xyshift=0.5,
            xshift_along_y=0.5,
            yshift_along_x=0.5,
            zshift_along_x=0.5,
            zshift_along_y=0.5,
            use_global=False,
        )
        plotter += axes
        
        
        # Return plotter for display
        return plotter
        
    else:
        # Use existing matplotlib code for 2D
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()
        
        ext = 0
        
        if not hasattr(model, 'ffs'):
            ffs = [model]
        else:
            ffs = model.ffs
        
        for i, ff in enumerate(ffs):
            Bx = ff.B[0, :].numpy()
            By = ff.B[1, :].numpy()
            
            try:
                if hasattr(ff, 'std'):
                    s = ff.std.item()
                    s_label = "std"
                    s_label = fr'{ff.short_name} {i} | {s_label} = {s:.3f}'
                elif hasattr(ff, 'sigma'):
                    s = ff.sigma.item()
                    s_label = "$\sigma$"
                    s_label = fr'{ff.short_name} {i} | {s_label} = {s:.3f}'
                elif hasattr(ff, 'K'):
                    s = ff.K
                    s_label = "K"
                    s_label = fr'{ff.short_name} {i} | {s_label} = {s:.0f}'
                elif hasattr(ff, 'r_in') and hasattr(ff, 'r_out'):
                    n_features = ff.B.shape[1]
                    s_label = fr'{ff.short_name} {i} | {n_features} pts'
                    
                ax.plot(Bx, By, 'o', fillstyle='none', label=s_label)
                ext = max(ext, np.max(np.abs(Bx)), np.max(np.abs(By)))
                
            except Exception as e:
                print(f"Could not plot layer {i}: {e}")
        
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_xlim(-ext*1.1, ext*1.1)
        ax.set_ylim(-ext*1.1, ext*1.1)
        ax.legend()
        
        return ax

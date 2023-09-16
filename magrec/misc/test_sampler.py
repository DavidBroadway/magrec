import torch
from magrec.misc.sampler import GridSampler, Sampler

def simple_sample_fn(num_points):
    """
    Sample points from a uniform interval between [0, 1).
    """
    return torch.rand(num_points)

def test_sampler():
    from magrec.misc.sampler import Sampler
    device = "cpu"  # Using CPU for simplicity
    cached_n_points = 20
    batch_n_points = 7
    
    # Create a sampler with regenerate_cache=False
    sampler_no_regen = Sampler(simple_sample_fn, batch_n_points, cached_n_points, regenerate_cache=False)
    
    batch_list_no_regen = [batch for _, batch in zip(range(3), sampler_no_regen.sample_points(device))]
    
    assert len(batch_list_no_regen) == 3, "Failed to produce the correct number of batches"
    assert len(batch_list_no_regen[0]) == 7, "First batch size is incorrect"
    assert len(batch_list_no_regen[1]) == 7, "Second batch size is incorrect"
    assert len(batch_list_no_regen[2]) == 6, "Third batch size is incorrect"
    
    # Create a sampler with regenerate_cache=True
    sampler_regen = Sampler(simple_sample_fn, batch_n_points, cached_n_points, regenerate_cache=True)
    
    batch_list_regen = [batch for _, batch in zip(range(4), sampler_regen.sample_points(device))]
    
    assert len(batch_list_regen) == 4, "Failed to produce the correct number of batches"
    assert len(batch_list_regen[0]) == 7, "First batch size is incorrect"
    assert len(batch_list_regen[1]) == 7, "Second batch size is incorrect"
    assert len(batch_list_regen[2]) == 6, "Third batch size is incorrect"
    assert len(batch_list_regen[3]) == 7, "Fourth batch size (after regeneration) is incorrect"
    
    print("All tests passed!")


def simple_sample_fn_2d(num_points):
    """
    Sample 2D points from a uniform interval between [0, 1) for both x and y.
    """
    return torch.rand(num_points, 2)


def test_sampler_visual():
    """Tests that two samplers (one with regenerate_cache=False and one with regenerate_cache=True) produce
    points as expected. Shows the plot with two subplots"""
    import matplotlib.pyplot as plt
        
    def simple_sample_fn(num_points):
        """
        Sample points from a uniform interval between [0, 1).
        """
        return torch.rand(num_points, 1)
    
    colors = ['r', 'g', 'b', 'y', 'm', 'c', 'k']  # Define more colors if needed
    
    fig = plt.figure(figsize=(8, 4))
    axs = fig.subplots(2, 1, sharex=True)
        
    device = "cpu"  # Using CPU for simplicity
    cached_n_points = 20
    batch_n_points = 7
    num_batches = 7
    num_batches_to_exhaust_cache = cached_n_points // batch_n_points + 1

    # Create and visualize a sampler with regenerate_cache=False
    sampler_no_regen = Sampler(simple_sample_fn, batch_n_points, cached_n_points, regenerate_cache=False)
    
    for i in range(num_batches):
        batch = sampler_no_regen.sample_points(device)
        axs[0].scatter(torch.ones(len(batch)) * (i % num_batches_to_exhaust_cache) + 0.1 * (i // num_batches_to_exhaust_cache), batch, color=colors[i])
        
    sampler_regen = Sampler(simple_sample_fn, batch_n_points, cached_n_points, regenerate_cache=True)
        
    for i in range(num_batches):
        batch = sampler_regen.sample_points(device)
        axs[1].scatter(torch.ones(len(batch)) * (i % num_batches_to_exhaust_cache) + 0.1 * (i // num_batches_to_exhaust_cache), batch, color=colors[i])
        
    # Add text to the figure explaining the expected result. That in non regeneration mode, the batches repeat when exhausted,
    # and with regeneration, they are newly regenerated when exhausted.
    fig.text(0.02, 0.7, 
            "This plot shows the sampled points for two samplers, one with regenerate_cache=False\n"
            "and one with regenerate_cache=True. The expected behavior is that the batches repeat\n"
            "when exhausted with no regeneration, and are newly regenerated when exhausted with\n"
            "regeneration.", 
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="aliceblue"))
    
    # Add labels, title, and a text box
    axs[0].set_ylabel('y')
    axs[1].set_ylabel('y')
    axs[1].set_xlabel('Batch number')
    axs[0].set_title('Non-regeneration mode')
    axs[1].set_title('Regeneration mode')
    axs[0].grid(True)  # Adds gridlines for better visualization
    axs[1].grid(True)  # Adds gridlines for better visualization
    plt.tight_layout()  # Adjusts the spacing to ensure everything fits
    
    plt.show()
    
    

def test_grid_sampler():
    from magrec.misc.sampler import GridSampler
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Assuming the function GridSampler.sample_grid is defined somewhere
    grid_points = GridSampler.sample_grid(50, 10, origin=[-2, -1], diagonal=[1, 1])

    # Create a sequence of normalized values between 0 and 1 based on the number of grid points
    num_points = grid_points.shape[0]
    colors = np.linspace(0, 1, num_points)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(grid_points[:, 0], grid_points[:, 1], marker='o', c=colors, cmap='viridis', s=10)  # Adjusted marker and size for clarity

    # Set the title and labels
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Grid Points Distribution')

    # Create an axes divider to manage the size and position of the text box
    divider = make_axes_locatable(ax)
    ax_text = divider.append_axes("top", size="20%", pad=0.3)

    # Place the text inside the new axes and hide the axis
    ax_text.text(0.5, 0.5, 
                 "This plot shows a regular grid distribution, sampled from a rectangle\n"
                 "with origin (-2, -1) and diagonal (1, 1). There are 50 points in the x-direction\n"
                 "and 10 points in the y-direction for every unit rectangle.", 
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="aliceblue"),
                 horizontalalignment='center', verticalalignment='center')
    ax_text.axis('off')

    # Add a colorbar and set the ticks to represent the point numbers
    cbar = plt.colorbar(ax.scatter(grid_points[:, 0], grid_points[:, 1], c=colors, cmap='viridis', s=10), ax=ax, orientation='vertical')
    cbar.set_label('Point Number')
    tick_locs = (np.linspace(0, 1, 6) * num_points).astype(int)
    cbar.set_ticks(np.linspace(0, 1, 6))
    cbar.set_ticklabels(tick_locs)

    ax.grid(True)  # Adds gridlines for better visualization
    plt.tight_layout()  # Adjusts the spacing to ensure everything fits
    plt.show()


def test_grid_sampler_reshape():
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    W, H = 50, 10  # Given dimensions
    
    # Assuming the function GridSampler.sample_grid is defined somewhere
    grid_points = GridSampler.sample_grid(W, H, origin=[-2, -1], diagonal=[1, 1])

    # Create a sequence of normalized values between 0 and 1 based on the number of grid points
    num_points = grid_points.shape[0]
    colors = torch.linspace(0, 1, num_points)

    # Reshape the grid_points
    reshaped_grid_points = torch.tensor(grid_points).reshape(2, W, H)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Extract x and y coordinates after reshaping
    x = reshaped_grid_points[0].numpy()
    y = reshaped_grid_points[1].numpy()
    
    # Extract colors in a grid format
    reshaped_colors = torch.tile(colors.reshape(H, W), (2, 1, 1))[0]
    
    ax.scatter(x, y, c=reshaped_colors, cmap='viridis', s=10)

    # Set the title and labels
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Reshaped Grid Points Distribution')

    # Create an axes divider to manage the size and position of the text box
    divider = make_axes_locatable(ax)
    ax_text = divider.append_axes("top", size="20%", pad=0.3)

    # Place the text inside the new axes and hide the axis
    ax_text.text(0.5, 0.5, 
                 "This plot shows a regular grid distribution after reshaping.", 
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="aliceblue"),
                 horizontalalignment='center', verticalalignment='center')
    ax_text.axis('off')

    # Add a colorbar and set the ticks to represent the point numbers
    cbar = plt.colorbar(ax.scatter(x, y, c=reshaped_colors, cmap='viridis', s=10), ax=ax, orientation='vertical')
    cbar.set_label('Point Number')
    tick_locs = (torch.linspace(0, 1, 6) * num_points).int().numpy()
    cbar.set_ticks(torch.linspace(0, 1, 6))
    cbar.set_ticklabels(tick_locs)

    ax.grid(True)
    plt.tight_layout()
    plt.show()


def test_grid_sampler_pts_to_grid():
    n_x = 3
    n_y = 4
    
    pts = GridSampler.sample_grid(n_x, n_y, origin=[0, 0], diagonal=[n_x-1, n_y-1])
    grid = GridSampler.pts_to_grid(pts, n_x, n_y)
    
    assert grid.shape == (2, n_x, n_y), "The shape of the grid is incorrect"
    
    # Check values
    for i in range(n_x):
        for j in range(n_y):
            x, y = grid[:, i, j]
            expected_x, expected_y = pts[i * n_y + j]
            assert x == expected_x and y == expected_y, f"Expected ({expected_x}, {expected_y}), but got ({x}, {y}) at ({i}, {j})"
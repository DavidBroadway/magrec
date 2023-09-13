import torch
from magrec.misc.sampler import GridSampler, Sampler

def simple_sample_fn(num_points):
    """
    Sample points from a uniform interval between [0, 1).
    """
    return torch.rand(num_points)

def test_sampler():
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
    import matplotlib.pyplot as plt

    # Assuming the function GridSampler.sample_grid is defined somewhere
    grid_points = GridSampler.sample_grid(50, 10, origin=[-2, -1], diagonal=[1, 1])
    plt.scatter(grid_points[:, 0], grid_points[:, 1], marker='o', c='blue', s=10)  # Adjusted marker and size for clarity

    # Adding labels, title, and a text box
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Grid Points Distribution')
    plt.text(-1.5, -0.5, 
            "This plot shows a regular grid distribution, sampled from a rectangle\n"
            "with origin (-2, -1) and diagonal (1, 1). There are 50 points in the x-direction\n"
            "and 10 points in the y-direction for every unit rectangle.", 
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="aliceblue"))

    plt.grid(True)  # Adds gridlines for better visualization
    plt.tight_layout()  # Adjusts the spacing to ensure everything fits
    plt.show()
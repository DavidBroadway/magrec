import os
import inspect
from PIL import Image
import torch
from torchvision.transforms import ToTensor
from torch.nn.functional import mse_loss as mse
import matplotlib.pyplot as plt

def auto_reference_plot(func):
    def wrapper(*args, **kwargs):
        # Determine the directory of the test function
        test_file_path = inspect.getfile(func)
        test_directory = os.path.dirname(test_file_path)

        # Create test_plots directory if it doesn't exist
        test_plots_dir = os.path.join(test_directory, 'test_plots')
        
        if not os.path.exists(test_plots_dir):
            os.makedirs(test_plots_dir)

        # Execute the plotting function
        fig = func(*args, **kwargs)

        # Define filenames
        filename = func.__name__
        output_path = os.path.join(test_plots_dir, f'{filename}.png')
        reference_path = os.path.join(test_plots_dir, f'{filename}_reference.png')

        # Save the plot
        fig.savefig(output_path)

        # If reference exists, compare
        if os.path.exists(reference_path):
            to_tensor = ToTensor()
            reference_image = to_tensor(Image.open(reference_path))
            output_image = to_tensor(Image.open(output_path))

            # Calculate MSE between the two images
            mse_value = mse(reference_image, output_image)
            
            # You can set a threshold for MSE to determine if images are "different enough"
            if mse_value > 0.002:  # Adjust this threshold based on your needs
                print(f"WARNING: The plot for {filename} deviates from the reference. MSE: {mse_value:.4f}")
            
            assert mse_value < 0.002
            

        # If no reference exists, save the plot as reference (optional)
        else:
            fig.savefig(reference_path)
            print(f"Reference for {filename} saved.")

        plt.close(fig)  # Close the figure to free up memory
    return wrapper

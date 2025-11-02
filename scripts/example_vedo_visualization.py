#!/usr/bin/env python3
"""
Example script demonstrating the PyVista vector field visualization.
This script creates sample magnetic field and current distribution data
and visualizes them using the new PyVista-based visualization function.
"""

import numpy as np
import torch
from plot_vector_fields import visualize_vector_fields

def create_sample_magnetic_field(W=64, H=64):
    """Create a sample 3D magnetic field with a dipole-like pattern."""
    x, y = np.meshgrid(np.linspace(-2, 2, W), np.linspace(-2, 2, H), indexing='ij')
    
    # Create a dipole-like magnetic field
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    
    # Avoid division by zero
    r = np.where(r < 0.1, 0.1, r)
    
    # Dipole field components (B_r and B_theta)
    B_r = np.cos(theta) / (r**2)
    B_theta = -np.sin(theta) / (r**2)
    
    # Convert to Cartesian coordinates
    B_x = B_r * np.cos(theta) - B_theta * np.sin(theta)
    B_y = B_r * np.sin(theta) + B_theta * np.cos(theta)
    B_z = np.zeros_like(B_x)  # No z-component for 2D dipole
    
    # Normalize and scale
    B_mag = np.sqrt(B_x**2 + B_y**2 + B_z**2)
    B_x = B_x / B_mag.max() * 2.0
    B_y = B_y / B_mag.max() * 2.0
    B_z = B_z / B_mag.max() * 2.0
    
    return np.stack([B_x, B_y, B_z])

def create_sample_current_distribution(W=64, H=64):
    """Create a sample 2D current distribution with a circular current loop."""
    x, y = np.meshgrid(np.linspace(-2, 2, W), np.linspace(-2, 2, H), indexing='ij')
    
    # Create a circular current loop
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    
    # Current density: circular flow around origin
    J_x = -np.sin(theta) * np.exp(-(r - 1.0)**2 / 0.5**2)
    J_y = np.cos(theta) * np.exp(-(r - 1.0)**2 / 0.5**2)
    
    # Normalize
    J_mag = np.sqrt(J_x**2 + J_y**2)
    J_x = J_x / J_mag.max() * 1.5
    J_y = J_y / J_mag.max() * 1.5
    
    return np.stack([J_x, J_y])

def main():
    """Main function to demonstrate the PyVista visualization."""
    print("Creating sample magnetic field and current distribution...")
    
    # Create sample data
    magnetic_field = create_sample_magnetic_field(64, 64)
    current_distribution = create_sample_current_distribution(64, 64)
    
    print("Magnetic field shape:", magnetic_field.shape)
    print("Current distribution shape:", current_distribution.shape)
    
    # Create 3D visualization
    print("Creating 3D visualization...")
    plotter_3d = visualize_vector_fields(
        magnetic_field=magnetic_field,
        current_distribution=current_distribution,
        z1=1.0,  # Magnetic field at z=1
        z0=0.0,  # Current distribution at z=0
        num_arrows=15,
        show_inset=True,
        inset_region=((20, 20), (44, 44)),  # Zoom into center region
        window_size=(1000, 800),
        interactive=True
    )
    
    # Create 2D visualization (projection)
    print("Creating 2D visualization...")
    plotter_2d = visualize_vector_fields(
        magnetic_field=magnetic_field,
        current_distribution=current_distribution,
        num_arrows=20,
        window_size=(1200, 600),
        interactive=True
    )
    
    print("Visualizations completed!")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Evaluation script for trained joint PhysNet-DCMNet model.

Features:
- Single-point energy/forces/dipole/ESP evaluation
- 3D ESP visualization at multiple radial surfaces
- Comparison of PhysNet (point charges) vs DCMNet (distributed multipoles)
- Beautiful matplotlib 3D rendering
"""

import sys
from pathlib import Path
import numpy as np
import pickle
import argparse

repo_root = Path(__file__).parent / "../../.."
sys.path.insert(0, str(repo_root.resolve()))

import jax
import jax.numpy as jnp
from ase import Atoms
from ase.visualize.plot import plot_atoms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from trainer import JointPhysNetDCMNet, prepare_batch_data
from mmml.dcmnet.dcmnet.electrostatics import calc_esp


def create_spherical_grid_mesh(center, radius, n_theta=100, n_phi=100):
    """
    Create a regular spherical mesh for smooth surface rendering.
    
    Parameters
    ----------
    center : array
        Center of sphere
    radius : float
        Radius of sphere
    n_theta : int
        Number of points in azimuthal direction
    n_phi : int
        Number of points in polar direction
        
    Returns
    -------
    tuple
        (x, y, z) mesh grids for surface plotting
    """
    theta = np.linspace(0, 2*np.pi, n_theta)
    phi = np.linspace(0, np.pi, n_phi)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    
    x = radius * np.sin(phi_grid) * np.cos(theta_grid) + center[0]
    y = radius * np.sin(phi_grid) * np.sin(theta_grid) + center[1]
    z = radius * np.cos(phi_grid) + center[2]
    
    return x, y, z, theta_grid, phi_grid


def create_spherical_grid(center, radius, n_points=500):
    """
    Create approximately uniform points on a sphere.
    
    Uses Fibonacci sphere algorithm for even distribution.
    """
    indices = np.arange(0, n_points, dtype=float) + 0.5
    
    phi = np.arccos(1 - 2*indices/n_points)
    theta = np.pi * (1 + 5**0.5) * indices
    
    x = radius * np.sin(phi) * np.cos(theta) + center[0]
    y = radius * np.sin(phi) * np.sin(theta) + center[1]
    z = radius * np.cos(phi) + center[2]
    
    return np.column_stack([x, y, z])


def evaluate_molecule(params, model, atoms, n_dcm=3, cutoff=10.0):
    """
    Evaluate energy, forces, dipole, charges, and ESP for a molecule.
    
    Parameters
    ----------
    params : dict
        Model parameters
    model : JointPhysNetDCMNet
        The joint model
    atoms : Atoms
        ASE Atoms object
    n_dcm : int
        Number of distributed multipoles per atom
    cutoff : float
        Cutoff distance for edge list construction
        
    Returns
    -------
    dict
        Results containing all predictions
    """
    # Prepare inputs
    positions = atoms.get_positions()
    atomic_numbers = atoms.get_atomic_numbers()
    n_atoms = len(atoms)
    
    # Compute edge indices manually (pairwise distances within cutoff)
    dst_list = []
    src_list = []
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i != j:
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < cutoff:
                    dst_list.append(i)
                    src_list.append(j)
    
    dst_idx = np.array(dst_list, dtype=np.int32)
    src_idx = np.array(src_list, dtype=np.int32)
    
    # Prepare batch data
    batch_segments = np.zeros(len(atoms), dtype=np.int32)
    batch_mask = np.ones(len(dst_idx), dtype=np.float32)
    atom_mask = np.ones(len(atoms), dtype=np.float32)
    
    # Run model
    output = model.apply(
        params,
        atomic_numbers=jnp.array(atomic_numbers),
        positions=jnp.array(positions),
        dst_idx=jnp.array(dst_idx),
        src_idx=jnp.array(src_idx),
        batch_segments=jnp.array(batch_segments),
        batch_size=1,
        batch_mask=jnp.array(batch_mask),
        atom_mask=jnp.array(atom_mask),
    )
    
    # Extract results (model returns batched outputs, extract first element)
    energy = float(output['energy'][0]) if output['energy'].ndim > 0 else float(output['energy'])
    forces = np.array(output['forces'][:n_atoms])
    dipole_physnet = np.array(output['dipoles'][0])
    charges_physnet = np.array(output['charges_as_mono'][:n_atoms])
    
    # DCMNet outputs
    mono_dist = np.array(output['mono_dist'][:n_atoms])  # (natoms, n_dcm)
    dipo_dist = np.array(output['dipo_dist'][:n_atoms])  # (natoms, n_dcm, 3)
    
    # Compute DCMNet dipole
    import ase.data
    masses = np.array([ase.data.atomic_masses[z] for z in atomic_numbers])
    com = np.sum(positions * masses[:, None], axis=0) / masses.sum()
    dipo_rel_com = dipo_dist - com[None, None, :]
    dipole_dcmnet = np.sum(mono_dist[..., None] * dipo_rel_com, axis=(0, 1))
    
    return {
        'energy': energy,
        'forces': forces,
        'dipole_physnet': dipole_physnet,
        'dipole_dcmnet': dipole_dcmnet,
        'charges_physnet': charges_physnet,
        'charges_dcmnet': mono_dist,
        'positions_dcmnet': dipo_dist,
        'positions': positions,
        'atomic_numbers': atomic_numbers,
    }


def compute_esp_on_sphere(charges, positions, sphere_points):
    """Compute ESP from point charges on a spherical surface."""
    # charges: (n_charges,), positions: (n_charges, 3), sphere_points: (n_points, 3)
    diff = sphere_points[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(diff, axis=2)
    # ESP in Hartree/e with Å→Bohr conversion
    esp = np.sum(charges[None, :] / (distances * 1.88973 + 1e-10), axis=1)
    return esp


def plot_esp_spheres(results, radii=[2.0, 3.0, 4.0, 5.0], n_theta=150, n_phi=150, save_path=None, gt_esp_data=None):
    """
    Create beautiful 3D ESP visualization at multiple radial surfaces with smooth shading.
    
    Parameters
    ----------
    results : dict
        Output from evaluate_molecule
    radii : list
        Radii (in Angstroms) for ESP spheres
    n_theta : int
        Mesh resolution in azimuthal direction (higher = smoother)
    n_phi : int
        Mesh resolution in polar direction (higher = smoother)
    save_path : Path, optional
        Where to save the plot
    gt_esp_data : dict, optional
        Ground truth ESP data with keys 'grid_points' and 'esp_values'
    """
    positions = results['positions']
    charges_physnet = results['charges_physnet']
    charges_dcmnet = results['charges_dcmnet']
    positions_dcmnet = results['positions_dcmnet']
    atomic_numbers = results['atomic_numbers']
    
    # Compute molecular center of mass
    import ase.data
    masses = np.array([ase.data.atomic_masses[z] for z in atomic_numbers])
    com = np.sum(positions * masses[:, None], axis=0) / masses.sum()
    
    # Calculate consistent axis limits for all plots
    max_radius = max(radii)
    axis_limit = max_radius * 1.1  # 10% padding
    
    # Determine number of rows (3 or 6 depending on GT data availability)
    n_rows = 6 if gt_esp_data is not None else 3
    fig_height = 6 * n_rows  # 6 inches per row
    
    # Create figure
    fig = plt.figure(figsize=(6*len(radii), fig_height))
    
    # Fixed color scale for ESP with better range
    esp_vmin = -0.015
    esp_vmax = 0.015
    
    # Use a better colormap
    cmap = plt.cm.RdBu_r
    
    # Store ESP data for difference calculation
    esp_physnet_all = []
    esp_dcmnet_all = []
    esp_gt_all = []
    mesh_data_all = []
    
    # Prepare ground truth ESP interpolation if available
    if gt_esp_data is not None:
        from scipy.interpolate import NearestNDInterpolator
        gt_grid_points = gt_esp_data['grid_points']  # (n_grid, 3)
        gt_esp_values = gt_esp_data['esp_values']    # (n_grid,)
        
        # Create interpolator for GT ESP
        gt_interpolator = NearestNDInterpolator(gt_grid_points, gt_esp_values)
        print(f"  Using ground truth ESP with {len(gt_esp_values)} grid points")
    
    for i, radius in enumerate(radii):
        # Create spherical mesh centered on COM
        x_mesh, y_mesh, z_mesh, theta_grid, phi_grid = create_spherical_grid_mesh(
            com, radius, n_theta=n_theta, n_phi=n_phi
        )
        
        # Convert mesh to points for ESP evaluation
        sphere_points = np.column_stack([x_mesh.flatten(), y_mesh.flatten(), z_mesh.flatten()])
        
        # Compute ESP from PhysNet charges
        esp_physnet = compute_esp_on_sphere(charges_physnet, positions, sphere_points)
        esp_physnet_mesh = esp_physnet.reshape(x_mesh.shape)
        
        # Flatten distributed charges for ESP calculation
        charges_dcmnet_flat = charges_dcmnet.flatten()
        positions_dcmnet_flat = positions_dcmnet.reshape(-1, 3)
        
        # Compute ESP from DCMNet distributed charges using calc_esp
        esp_dcmnet = calc_esp(
            jnp.array(positions_dcmnet_flat),
            jnp.array(charges_dcmnet_flat),
            jnp.array(sphere_points)
        )
        esp_dcmnet = np.array(esp_dcmnet)
        esp_dcmnet_mesh = esp_dcmnet.reshape(x_mesh.shape)
        
        # Interpolate ground truth ESP if available
        if gt_esp_data is not None:
            esp_gt = gt_interpolator(sphere_points)
            esp_gt_mesh = esp_gt.reshape(x_mesh.shape)
        else:
            esp_gt_mesh = None
        
        # Store for difference calculation
        esp_physnet_all.append(esp_physnet_mesh)
        esp_dcmnet_all.append(esp_dcmnet_mesh)
        esp_gt_all.append(esp_gt_mesh)
        mesh_data_all.append((x_mesh, y_mesh, z_mesh))
        
        # === ROW 1: PhysNet ESP (point charges at atoms) ===
        ax = fig.add_subplot(n_rows, len(radii), i+1, projection='3d')
        
        # Normalize colors for better visualization
        norm = plt.Normalize(vmin=esp_vmin, vmax=esp_vmax)
        colors = cmap(norm(esp_physnet_mesh))
        
        # Plot surface with smooth shading
        surf = ax.plot_surface(x_mesh, y_mesh, z_mesh, facecolors=colors,
                              rstride=1, cstride=1, linewidth=0, 
                              antialiased=True, shade=True, alpha=0.95)
        
        # Add atoms with glow effect
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                  c='#FFD700', s=500, marker='o', edgecolors='yellow', linewidths=4,
                  alpha=1.0, zorder=100)
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                  c='black', s=200, marker='o', edgecolors='none',
                  alpha=1.0, zorder=101)
        
        # Label atoms with better styling
        for j, (pos, z) in enumerate(zip(positions, atomic_numbers)):
            ax.text(pos[0], pos[1], pos[2], f'{int(z)}', fontsize=12,
                   ha='center', va='center', color='white', weight='bold',
                   zorder=102)
        
        ax.set_xlabel('X (Å)', fontsize=11, weight='bold')
        ax.set_ylabel('Y (Å)', fontsize=11, weight='bold')
        ax.set_zlabel('Z (Å)', fontsize=11, weight='bold')
        ax.set_title(f'PhysNet ESP @ r = {radius:.1f} Å', fontsize=13, weight='bold', pad=10)
        ax.view_init(elev=25, azim=45)
        ax.set_box_aspect([1,1,1])
        
        # Set consistent axis limits
        ax.set_xlim([com[0] - axis_limit, com[0] + axis_limit])
        ax.set_ylim([com[1] - axis_limit, com[1] + axis_limit])
        ax.set_zlim([com[2] - axis_limit, com[2] + axis_limit])
        
        # Remove grid for cleaner look
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('lightgray')
        ax.yaxis.pane.set_edgecolor('lightgray')
        ax.zaxis.pane.set_edgecolor('lightgray')
        ax.xaxis.pane.set_alpha(0.3)
        ax.yaxis.pane.set_alpha(0.3)
        ax.zaxis.pane.set_alpha(0.3)
        
        if i == len(radii) - 1:
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, label='ESP (Ha/e)', shrink=0.7, pad=0.1)
            cbar.ax.tick_params(labelsize=10)
        
        # === ROW 2: DCMNet ESP (distributed multipoles) ===
        ax = fig.add_subplot(n_rows, len(radii), len(radii) + i + 1, projection='3d')
        
        # Normalize colors for better visualization
        colors = cmap(norm(esp_dcmnet_mesh))
        
        # Plot surface with smooth shading
        surf = ax.plot_surface(x_mesh, y_mesh, z_mesh, facecolors=colors,
                              rstride=1, cstride=1, linewidth=0,
                              antialiased=True, shade=True, alpha=0.95)
        
        # Add atoms with glow effect (cyan theme for DCMNet)
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                  c='#00FFFF', s=500, marker='o', edgecolors='cyan', linewidths=4,
                  alpha=1.0, zorder=100)
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                  c='black', s=200, marker='o', edgecolors='none',
                  alpha=1.0, zorder=101)
        
        # Add distributed charges (only significant ones)
        significant = np.abs(charges_dcmnet_flat) > 0.05
        if np.any(significant):
            # Color by charge magnitude
            charge_colors = charges_dcmnet_flat[significant]
            charge_norm = plt.Normalize(vmin=-0.5, vmax=0.5)
            ax.scatter(positions_dcmnet_flat[significant, 0],
                      positions_dcmnet_flat[significant, 1],
                      positions_dcmnet_flat[significant, 2],
                      c=charge_colors, cmap='RdBu_r', norm=charge_norm,
                      s=100, marker='^', edgecolors='white', linewidths=2,
                      alpha=0.9, zorder=99)
        
        # Label atoms with better styling
        for j, (pos, z) in enumerate(zip(positions, atomic_numbers)):
            ax.text(pos[0], pos[1], pos[2], f'{int(z)}', fontsize=12,
                   ha='center', va='center', color='white', weight='bold',
                   zorder=102)
        
        ax.set_xlabel('X (Å)', fontsize=11, weight='bold')
        ax.set_ylabel('Y (Å)', fontsize=11, weight='bold')
        ax.set_zlabel('Z (Å)', fontsize=11, weight='bold')
        ax.set_title(f'DCMNet ESP @ r = {radius:.1f} Å', fontsize=13, weight='bold', pad=10)
        ax.view_init(elev=25, azim=45)
        ax.set_box_aspect([1,1,1])
        
        # Set consistent axis limits
        ax.set_xlim([com[0] - axis_limit, com[0] + axis_limit])
        ax.set_ylim([com[1] - axis_limit, com[1] + axis_limit])
        ax.set_zlim([com[2] - axis_limit, com[2] + axis_limit])
        
        # Remove grid for cleaner look
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('lightgray')
        ax.yaxis.pane.set_edgecolor('lightgray')
        ax.zaxis.pane.set_edgecolor('lightgray')
        ax.xaxis.pane.set_alpha(0.3)
        ax.yaxis.pane.set_alpha(0.3)
        ax.zaxis.pane.set_alpha(0.3)
        
        if i == len(radii) - 1:
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, label='ESP (Ha/e)', shrink=0.7, pad=0.1)
            cbar.ax.tick_params(labelsize=10)
    
    # === ROW 3: Difference (PhysNet - DCMNet) ===
    # Color scale for differences (symmetric around zero)
    diff_vmin = -0.01
    diff_vmax = 0.01
    diff_norm = plt.Normalize(vmin=diff_vmin, vmax=diff_vmax)
    
    for i, radius in enumerate(radii):
        ax = fig.add_subplot(3, len(radii), 2*len(radii) + i + 1, projection='3d')
        
        x_mesh, y_mesh, z_mesh = mesh_data_all[i]
        esp_diff_mesh = esp_physnet_all[i] - esp_dcmnet_all[i]
        
        # Normalize colors
        colors = cmap(diff_norm(esp_diff_mesh))
        
        # Plot surface
        surf = ax.plot_surface(x_mesh, y_mesh, z_mesh, facecolors=colors,
                              rstride=1, cstride=1, linewidth=0,
                              antialiased=True, shade=True, alpha=0.95)
        
        # Add atoms (magenta theme for difference)
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                  c='#FF00FF', s=500, marker='o', edgecolors='magenta', linewidths=4,
                  alpha=1.0, zorder=100)
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                  c='black', s=200, marker='o', edgecolors='none',
                  alpha=1.0, zorder=101)
        
        # Label atoms
        for j, (pos, z) in enumerate(zip(positions, atomic_numbers)):
            ax.text(pos[0], pos[1], pos[2], f'{int(z)}', fontsize=12,
                   ha='center', va='center', color='white', weight='bold',
                   zorder=102)
        
        ax.set_xlabel('X (Å)', fontsize=11, weight='bold')
        ax.set_ylabel('Y (Å)', fontsize=11, weight='bold')
        ax.set_zlabel('Z (Å)', fontsize=11, weight='bold')
        ax.set_title(f'Difference (PhysNet - DCMNet) @ r = {radius:.1f} Å', 
                    fontsize=13, weight='bold', pad=10)
        ax.view_init(elev=25, azim=45)
        ax.set_box_aspect([1,1,1])
        
        # Set consistent axis limits
        ax.set_xlim([com[0] - axis_limit, com[0] + axis_limit])
        ax.set_ylim([com[1] - axis_limit, com[1] + axis_limit])
        ax.set_zlim([com[2] - axis_limit, com[2] + axis_limit])
        
        # Remove grid
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('lightgray')
        ax.yaxis.pane.set_edgecolor('lightgray')
        ax.zaxis.pane.set_edgecolor('lightgray')
        ax.xaxis.pane.set_alpha(0.3)
        ax.yaxis.pane.set_alpha(0.3)
        ax.zaxis.pane.set_alpha(0.3)
        
        if i == len(radii) - 1:
            # Add colorbar for difference
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=diff_norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, label='ESP Difference (Ha/e)', shrink=0.7, pad=0.1)
            cbar.ax.tick_params(labelsize=10)
    
    # === Additional rows if GT data available ===
    if gt_esp_data is not None:
        # ROW 4: Ground Truth ESP
        for i, radius in enumerate(radii):
            ax = fig.add_subplot(n_rows, len(radii), 3*len(radii) + i + 1, projection='3d')
            
            x_mesh, y_mesh, z_mesh = mesh_data_all[i]
            esp_gt_mesh = esp_gt_all[i]
            
            # Normalize colors
            colors = cmap(norm(esp_gt_mesh))
            
            # Plot surface
            surf = ax.plot_surface(x_mesh, y_mesh, z_mesh, facecolors=colors,
                                  rstride=1, cstride=1, linewidth=0,
                                  antialiased=True, shade=True, alpha=0.95)
            
            # Add atoms (green theme for GT)
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                      c='#00FF00', s=500, marker='o', edgecolors='lime', linewidths=4,
                      alpha=1.0, zorder=100)
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                      c='black', s=200, marker='o', edgecolors='none',
                      alpha=1.0, zorder=101)
            
            # Label atoms
            for j, (pos, z) in enumerate(zip(positions, atomic_numbers)):
                ax.text(pos[0], pos[1], pos[2], f'{int(z)}', fontsize=12,
                       ha='center', va='center', color='white', weight='bold',
                       zorder=102)
            
            ax.set_xlabel('X (Å)', fontsize=11, weight='bold')
            ax.set_ylabel('Y (Å)', fontsize=11, weight='bold')
            ax.set_zlabel('Z (Å)', fontsize=11, weight='bold')
            ax.set_title(f'Ground Truth ESP @ r = {radius:.1f} Å', 
                        fontsize=13, weight='bold', pad=10)
            ax.view_init(elev=25, azim=45)
            ax.set_box_aspect([1,1,1])
            
            # Set consistent axis limits
            ax.set_xlim([com[0] - axis_limit, com[0] + axis_limit])
            ax.set_ylim([com[1] - axis_limit, com[1] + axis_limit])
            ax.set_zlim([com[2] - axis_limit, com[2] + axis_limit])
            
            # Remove grid
            ax.grid(False)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('lightgray')
            ax.yaxis.pane.set_edgecolor('lightgray')
            ax.zaxis.pane.set_edgecolor('lightgray')
            ax.xaxis.pane.set_alpha(0.3)
            ax.yaxis.pane.set_alpha(0.3)
            ax.zaxis.pane.set_alpha(0.3)
            
            if i == len(radii) - 1:
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, label='ESP (Ha/e)', shrink=0.7, pad=0.1)
                cbar.ax.tick_params(labelsize=10)
        
        # ROW 5: PhysNet Error (PhysNet - GT)
        for i, radius in enumerate(radii):
            ax = fig.add_subplot(n_rows, len(radii), 4*len(radii) + i + 1, projection='3d')
            
            x_mesh, y_mesh, z_mesh = mesh_data_all[i]
            esp_error_mesh = esp_physnet_all[i] - esp_gt_all[i]
            
            # Normalize colors
            colors = cmap(diff_norm(esp_error_mesh))
            
            # Plot surface
            surf = ax.plot_surface(x_mesh, y_mesh, z_mesh, facecolors=colors,
                                  rstride=1, cstride=1, linewidth=0,
                                  antialiased=True, shade=True, alpha=0.95)
            
            # Add atoms (orange theme for PhysNet error)
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                      c='#FFA500', s=500, marker='o', edgecolors='orange', linewidths=4,
                      alpha=1.0, zorder=100)
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                      c='black', s=200, marker='o', edgecolors='none',
                      alpha=1.0, zorder=101)
            
            # Label atoms
            for j, (pos, z) in enumerate(zip(positions, atomic_numbers)):
                ax.text(pos[0], pos[1], pos[2], f'{int(z)}', fontsize=12,
                       ha='center', va='center', color='white', weight='bold',
                       zorder=102)
            
            ax.set_xlabel('X (Å)', fontsize=11, weight='bold')
            ax.set_ylabel('Y (Å)', fontsize=11, weight='bold')
            ax.set_zlabel('Z (Å)', fontsize=11, weight='bold')
            ax.set_title(f'PhysNet Error (pred - GT) @ r = {radius:.1f} Å', 
                        fontsize=13, weight='bold', pad=10)
            ax.view_init(elev=25, azim=45)
            ax.set_box_aspect([1,1,1])
            
            # Set consistent axis limits
            ax.set_xlim([com[0] - axis_limit, com[0] + axis_limit])
            ax.set_ylim([com[1] - axis_limit, com[1] + axis_limit])
            ax.set_zlim([com[2] - axis_limit, com[2] + axis_limit])
            
            # Remove grid
            ax.grid(False)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('lightgray')
            ax.yaxis.pane.set_edgecolor('lightgray')
            ax.zaxis.pane.set_edgecolor('lightgray')
            ax.xaxis.pane.set_alpha(0.3)
            ax.yaxis.pane.set_alpha(0.3)
            ax.zaxis.pane.set_alpha(0.3)
            
            if i == len(radii) - 1:
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=diff_norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, label='ESP Error (Ha/e)', shrink=0.7, pad=0.1)
                cbar.ax.tick_params(labelsize=10)
        
        # ROW 6: DCMNet Error (DCMNet - GT)
        for i, radius in enumerate(radii):
            ax = fig.add_subplot(n_rows, len(radii), 5*len(radii) + i + 1, projection='3d')
            
            x_mesh, y_mesh, z_mesh = mesh_data_all[i]
            esp_error_mesh = esp_dcmnet_all[i] - esp_gt_all[i]
            
            # Normalize colors
            colors = cmap(diff_norm(esp_error_mesh))
            
            # Plot surface
            surf = ax.plot_surface(x_mesh, y_mesh, z_mesh, facecolors=colors,
                                  rstride=1, cstride=1, linewidth=0,
                                  antialiased=True, shade=True, alpha=0.95)
            
            # Add atoms (teal theme for DCMNet error)
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                      c='#008080', s=500, marker='o', edgecolors='teal', linewidths=4,
                      alpha=1.0, zorder=100)
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                      c='black', s=200, marker='o', edgecolors='none',
                      alpha=1.0, zorder=101)
            
            # Label atoms
            for j, (pos, z) in enumerate(zip(positions, atomic_numbers)):
                ax.text(pos[0], pos[1], pos[2], f'{int(z)}', fontsize=12,
                       ha='center', va='center', color='white', weight='bold',
                       zorder=102)
            
            ax.set_xlabel('X (Å)', fontsize=11, weight='bold')
            ax.set_ylabel('Y (Å)', fontsize=11, weight='bold')
            ax.set_zlabel('Z (Å)', fontsize=11, weight='bold')
            ax.set_title(f'DCMNet Error (pred - GT) @ r = {radius:.1f} Å', 
                        fontsize=13, weight='bold', pad=10)
            ax.view_init(elev=25, azim=45)
            ax.set_box_aspect([1,1,1])
            
            # Set consistent axis limits
            ax.set_xlim([com[0] - axis_limit, com[0] + axis_limit])
            ax.set_ylim([com[1] - axis_limit, com[1] + axis_limit])
            ax.set_zlim([com[2] - axis_limit, com[2] + axis_limit])
            
            # Remove grid
            ax.grid(False)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('lightgray')
            ax.yaxis.pane.set_edgecolor('lightgray')
            ax.zaxis.pane.set_edgecolor('lightgray')
            ax.xaxis.pane.set_alpha(0.3)
            ax.yaxis.pane.set_alpha(0.3)
            ax.zaxis.pane.set_alpha(0.3)
            
            if i == len(radii) - 1:
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=diff_norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, label='ESP Error (Ha/e)', shrink=0.7, pad=0.1)
                cbar.ax.tick_params(labelsize=10)
    
    plt.suptitle('Electrostatic Potential on Spherical Surfaces', 
                 fontsize=18, weight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Saved ESP sphere visualization: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained joint model')
    parser.add_argument('--checkpoint', type=Path, required=True,
                       help='Path to checkpoint directory')
    parser.add_argument('--molecule', type=str, default='CO2',
                       help='Molecule formula (e.g., CO2, H2O)')
    parser.add_argument('--geometry', type=str, default=None,
                       help='XYZ file with geometry (optional, uses default if not provided)')
    parser.add_argument('--esp-data', type=Path, default=None,
                       help='Path to validation ESP NPZ file for ground truth comparison')
    parser.add_argument('--esp-index', type=int, default=0,
                       help='Index of molecule in ESP dataset (default: 0)')
    parser.add_argument('--radii', type=float, nargs='+', default=[2.0, 3.0, 4.0, 5.0],
                       help='Radii (Å) for ESP spherical surfaces')
    parser.add_argument('--mesh-resolution', type=int, default=150,
                       help='Mesh resolution for spheres (higher = smoother, 150 recommended)')
    parser.add_argument('--output-dir', type=Path, default=None,
                       help='Output directory for plots')
    args = parser.parse_args()
    
    print("="*70)
    print("Joint PhysNet-DCMNet Model Evaluation")
    print("="*70)
    
    # Load checkpoint
    print(f"\n1. Loading checkpoint from {args.checkpoint}")
    with open(args.checkpoint / 'best_params.pkl', 'rb') as f:
        params = pickle.load(f)
    
    with open(args.checkpoint / 'model_config.pkl', 'rb') as f:
        config = pickle.load(f)
    
    print(f"✅ Loaded {sum(x.size for x in jax.tree_util.tree_leaves(params)):,} parameters")
    
    # Create model
    model = JointPhysNetDCMNet(
        physnet_config=config['physnet_config'],
        dcmnet_config=config['dcmnet_config'],
        mix_coulomb_energy=config.get('mix_coulomb_energy', False)
    )
    
    n_dcm = config['dcmnet_config']['n_dcm']
    
    # Load or create molecule
    print(f"\n2. Setting up molecule: {args.molecule}")
    
    if args.geometry:
        from ase.io import read
        atoms = read(args.geometry)
        print(f"✅ Loaded geometry from {args.geometry}")
    else:
        # Create default molecule
        if args.molecule.upper() == 'CO2':
            atoms = Atoms('CO2', positions=[
                [0.0, 0.0, 0.0],      # C
                [0.0, 0.0, 1.16],     # O
                [0.0, 0.0, -1.16],    # O
            ])
        elif args.molecule.upper() == 'H2O':
            atoms = Atoms('H2O', positions=[
                [0.0, 0.0, 0.0],           # O
                [0.0, 0.757, 0.586],       # H
                [0.0, -0.757, 0.586],      # H
            ])
        else:
            raise ValueError(f"Unknown molecule: {args.molecule}. Provide --geometry XYZ file.")
        
        print(f"✅ Using default {args.molecule} geometry")
    
    print(f"\nMolecule:")
    print(f"  Formula: {atoms.get_chemical_formula()}")
    print(f"  Atoms: {len(atoms)}")
    print(f"  Positions (Å):")
    for i, (pos, z) in enumerate(zip(atoms.get_positions(), atoms.get_atomic_numbers())):
        print(f"    {i}: {z:2d} {pos}")
    
    # Evaluate
    print(f"\n3. Evaluating model...")
    results = evaluate_molecule(params, model, atoms, n_dcm=n_dcm)
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    
    print(f"\nEnergy: {results['energy']:.6f} eV")
    
    print(f"\nForces (eV/Å):")
    for i, f in enumerate(results['forces']):
        print(f"  Atom {i}: [{f[0]:10.6f}, {f[1]:10.6f}, {f[2]:10.6f}]")
    
    print(f"\nDipole moment:")
    d_phys = results['dipole_physnet']
    d_dcm = results['dipole_dcmnet']
    mag_phys = np.linalg.norm(d_phys)
    mag_dcm = np.linalg.norm(d_dcm)
    print(f"  PhysNet: [{d_phys[0]:.4f}, {d_phys[1]:.4f}, {d_phys[2]:.4f}] e·Å  (|D| = {mag_phys:.4f} e·Å = {mag_phys*4.8032:.4f} D)")
    print(f"  DCMNet:  [{d_dcm[0]:.4f}, {d_dcm[1]:.4f}, {d_dcm[2]:.4f}] e·Å  (|D| = {mag_dcm:.4f} e·Å = {mag_dcm*4.8032:.4f} D)")
    
    print(f"\nAtomic charges (PhysNet):")
    for i, q in enumerate(results['charges_physnet']):
        print(f"  Atom {i} (Z={atoms.get_atomic_numbers()[i]}): {q:.4f} e")
    print(f"  Total: {results['charges_physnet'].sum():.6f} e")
    
    print(f"\nDistributed charges (DCMNet):")
    for i in range(len(atoms)):
        q_dist = results['charges_dcmnet'][i]
        print(f"  Atom {i} (Z={atoms.get_atomic_numbers()[i]}): {q_dist.sum():.4f} e (from {len(q_dist)} distributed charges)")
    print(f"  Total: {results['charges_dcmnet'].sum():.6f} e")
    
    # Load ground truth ESP if provided
    gt_esp_data = None
    if args.esp_data:
        print(f"\n4. Loading ground truth ESP data from {args.esp_data}...")
        esp_npz = np.load(args.esp_data)
        
        # Extract data for the specified molecule index
        vdw_surface = esp_npz['vdw_surface'][args.esp_index]  # (n_grid, 3)
        esp_values = esp_npz['esp'][args.esp_index]           # (n_grid,)
        
        # Align to same coordinate frame as predicted (center on COM)
        import ase.data
        masses = np.array([ase.data.atomic_masses[z] for z in atoms.get_atomic_numbers()])
        positions_mol = atoms.get_positions()
        com = np.sum(positions_mol * masses[:, None], axis=0) / masses.sum()
        
        # Center ground truth grid points
        vdw_com = vdw_surface.mean(axis=0)
        vdw_surface_aligned = vdw_surface - vdw_com + com
        
        gt_esp_data = {
            'grid_points': vdw_surface_aligned,
            'esp_values': esp_values
        }
        
        print(f"✅ Loaded ground truth ESP: {len(esp_values)} grid points")
        print(f"   GT ESP range: [{esp_values.min():.6f}, {esp_values.max():.6f}] Ha/e")
    
    # Create ESP visualization
    print(f"\n5. Creating ESP visualization at {len(args.radii)} radii...")
    print(f"   Using mesh resolution: {args.mesh_resolution}×{args.mesh_resolution} ({args.mesh_resolution**2:,} points per sphere)")
    
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        esp_plot_path = args.output_dir / f'esp_spheres_{args.molecule.lower()}.png'
    else:
        esp_plot_path = Path(f'esp_spheres_{args.molecule.lower()}.png')
    
    plot_esp_spheres(results, radii=args.radii, n_theta=args.mesh_resolution, 
                     n_phi=args.mesh_resolution, save_path=esp_plot_path, 
                     gt_esp_data=gt_esp_data)
    
    # Create molecule visualization
    print(f"\n6. Creating molecule visualization...")
    
    # Extract positions and atomic numbers from results for plotting
    positions = results['positions']
    atomic_numbers = results['atomic_numbers']
    
    fig = plt.figure(figsize=(18, 6))
    
    # ASE plot
    ax = fig.add_subplot(131)
    plot_atoms(atoms, ax=ax, radii=0.5, rotation=('90x,0y,0z'))
    ax.set_title(f'{args.molecule} Molecule (ASE)', fontsize=14, weight='bold')
    
    # 3D with PhysNet charges
    ax = fig.add_subplot(132, projection='3d')
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
              c=results['charges_physnet'], cmap='RdBu_r', s=500,
              vmin=-1.0, vmax=1.0, edgecolors='black', linewidths=3, alpha=0.9)
    for i, (pos, z, q) in enumerate(zip(positions, atomic_numbers, results['charges_physnet'])):
        ax.text(pos[0], pos[1], pos[2], f'{int(z)}\n{q:.2f}e', fontsize=10,
               ha='center', va='center', color='white', weight='bold',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.7))
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title('PhysNet Point Charges', fontsize=14, weight='bold')
    ax.view_init(elev=35, azim=45)
    ax.set_box_aspect([1,1,1])
    
    # 3D with DCMNet distributed charges
    ax = fig.add_subplot(133, projection='3d')
    # Plot atoms
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
              c='black', s=300, marker='o', edgecolors='yellow', linewidths=3, alpha=1.0)
    # Plot distributed charges
    charges_flat = results['charges_dcmnet'].flatten()
    positions_flat = results['positions_dcmnet'].reshape(-1, 3)
    significant = np.abs(charges_flat) > 0.05
    if np.any(significant):
        ax.scatter(positions_flat[significant, 0],
                  positions_flat[significant, 1],
                  positions_flat[significant, 2],
                  c=charges_flat[significant], cmap='RdBu_r', s=100,
                  marker='^', edgecolors='white', linewidths=2,
                  vmin=-0.5, vmax=0.5, alpha=0.9)
    # Label atoms
    for i, (pos, z) in enumerate(zip(positions, atomic_numbers)):
        ax.text(pos[0], pos[1], pos[2], f'  {int(z)}', fontsize=10,
               color='black', weight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title(f'DCMNet Distributed Charges\n({n_dcm} per atom)', fontsize=14, weight='bold')
    ax.view_init(elev=35, azim=45)
    ax.set_box_aspect([1,1,1])
    
    plt.suptitle(f'{args.molecule} - Charge Distribution', fontsize=16, weight='bold')
    plt.tight_layout()
    
    if args.output_dir:
        mol_plot_path = args.output_dir / f'molecule_{args.molecule.lower()}.png'
    else:
        mol_plot_path = Path(f'molecule_{args.molecule.lower()}.png')
    
    plt.savefig(mol_plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved molecule visualization: {mol_plot_path}")
    
    print(f"\n{'='*70}")
    print("✅ EVALUATION COMPLETE")
    print(f"{'='*70}")
    
    if args.output_dir:
        print(f"\nAll outputs saved to: {args.output_dir}")
    
    return results


if __name__ == '__main__':
    main()


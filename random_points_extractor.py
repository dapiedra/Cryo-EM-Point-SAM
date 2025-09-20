#!/usr/bin/env python3
"""
Script to extract 3 random points from a PLY file and save them as red points in a new PLY file.
"""

import os
import random
import argparse
import numpy as np
from pc_sam.ply_utils import load_ply, save_ply


def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two 3D points."""
    return np.sqrt(np.sum((p1 - p2) ** 2))


def select_clustered_points(points, num_points=3, cluster_radius=None):
    """
    Select points that are close to each other.
    
    Args:
        points (np.array): Array of 3D points with shape [N, 3]
        num_points (int): Number of points to select
        cluster_radius (float): Maximum distance between points in cluster.
                               If None, will be estimated automatically.
    
    Returns:
        np.array: Selected clustered points
        list: Indices of selected points
    """
    if len(points) < num_points:
        return points, list(range(len(points)))
    
    # If cluster_radius is not specified, estimate it based on point cloud density
    if cluster_radius is None:
        # Sample some points to estimate average nearest neighbor distance
        sample_size = min(1000, len(points))
        sample_indices = np.random.choice(len(points), sample_size, replace=False)
        sample_points = points[sample_indices]
        
        # Calculate average distance to nearest neighbor using vectorized operations
        distances_matrix = np.sqrt(np.sum((sample_points[:100, np.newaxis, :] - sample_points[np.newaxis, :, :]) ** 2, axis=2))
        # Set diagonal to inf to exclude self-distances
        np.fill_diagonal(distances_matrix[:100, :100], np.inf)
        min_distances = np.min(distances_matrix[:100], axis=1)
        
        if len(min_distances) > 0:
            avg_min_dist = np.mean(min_distances[min_distances < np.inf])
            cluster_radius = avg_min_dist * 3  # Use 3x average nearest neighbor distance
        else:
            cluster_radius = 0.1  # Fallback value
    
    print(f"Using cluster radius: {cluster_radius:.6f}")
    
    best_cluster = None
    best_indices = None
    max_attempts = 50  # Reduced attempts for efficiency
    
    for attempt in range(max_attempts):
        # Start with a random seed point
        seed_idx = random.randint(0, len(points) - 1)
        seed_point = points[seed_idx]
        
        # Find all points within cluster_radius of the seed using vectorized operations
        distances = np.sqrt(np.sum((points - seed_point) ** 2, axis=1))
        candidate_indices = np.where(distances <= cluster_radius)[0].tolist()
        
        # If we have enough candidates, select the required number
        if len(candidate_indices) >= num_points:
            selected_indices = random.sample(candidate_indices, num_points)
            selected_points = points[selected_indices]
            
            # Calculate cluster compactness (average pairwise distance) using vectorized operations
            if len(selected_points) > 1:
                pairwise_distances = np.sqrt(np.sum((selected_points[:, np.newaxis, :] - selected_points[np.newaxis, :, :]) ** 2, axis=2))
                # Get upper triangular part (excluding diagonal)
                upper_tri_mask = np.triu(np.ones_like(pairwise_distances, dtype=bool), k=1)
                avg_distance = np.mean(pairwise_distances[upper_tri_mask])
            else:
                avg_distance = 0
            
            # Keep the most compact cluster found so far
            if best_cluster is None or avg_distance < best_cluster:
                best_cluster = avg_distance
                best_indices = selected_indices
    
    if best_indices is None:
        print(f"Warning: Could not find {num_points} points within cluster radius {cluster_radius:.6f}")
        print("Falling back to closest points to a random seed...")
        
        # Fallback: select closest points to a random seed
        seed_idx = random.randint(0, len(points) - 1)
        seed_point = points[seed_idx]
        
        # Calculate distances to all points using vectorized operations
        distances = np.sqrt(np.sum((points - seed_point) ** 2, axis=1))
        closest_indices = np.argsort(distances)[:num_points]
        best_indices = closest_indices.tolist()
    
    selected_points = points[best_indices]
    
    # Print cluster statistics
    if len(selected_points) > 1:
        pairwise_distances = np.sqrt(np.sum((selected_points[:, np.newaxis, :] - selected_points[np.newaxis, :, :]) ** 2, axis=2))
        upper_tri_mask = np.triu(np.ones_like(pairwise_distances, dtype=bool), k=1)
        distances = pairwise_distances[upper_tri_mask]
        
        print(f"Cluster statistics:")
        print(f"  Average pairwise distance: {np.mean(distances):.6f}")
        print(f"  Max pairwise distance: {np.max(distances):.6f}")
        print(f"  Min pairwise distance: {np.min(distances):.6f}")
    
    return selected_points, best_indices


def extract_random_points(input_ply_path, output_ply_path, num_points=3, clustered=False, cluster_radius=None):
    """
    Extract random points from a PLY file and save them as red points.
    
    Args:
        input_ply_path (str): Path to the input PLY file
        output_ply_path (str): Path to save the output PLY file
        num_points (int): Number of random points to extract (default: 3)
        clustered (bool): If True, select points close to each other
        cluster_radius (float): Maximum distance for clustered points (auto-estimated if None)
    """
    print(f"Loading PLY file: {input_ply_path}")
    
    # Load the original PLY file
    # The load_ply function returns points with shape [N, 6] where columns are [x, y, z, r, g, b]
    points_data = load_ply(input_ply_path)
    
    print(f"Loaded {points_data.shape[0]} points from {input_ply_path}")
    
    # Extract only the xyz coordinates (first 3 columns)
    points_xyz = points_data[:, :3]
    
    # Select points based on the clustering option
    if clustered:
        print(f"Selecting {num_points} clustered points...")
        selected_points, selected_indices = select_clustered_points(points_xyz, num_points, cluster_radius)
    else:
        print(f"Selecting {num_points} random points...")
        # Randomly select the specified number of points
        if points_data.shape[0] < num_points:
            print(f"Warning: PLY file only has {points_data.shape[0]} points, selecting all of them.")
            selected_indices = list(range(points_data.shape[0]))
        else:
            selected_indices = random.sample(range(points_data.shape[0]), num_points)
        
        selected_points = points_xyz[selected_indices]
    
    print(f"Selected {len(selected_points)} points:")
    for i, point in enumerate(selected_points):
        print(f"  Point {i+1}: ({point[0]:.6f}, {point[1]:.6f}, {point[2]:.6f})")
    
    # Create red color for all selected points (RGB values in range [0, 1])
    red_colors = np.array([[1.0, 0.0, 0.0]] * len(selected_points))
    
    # Save the selected points as red points
    save_ply(output_ply_path, selected_points, red_colors)
    
    print(f"Saved {len(selected_points)} red points to: {output_ply_path}")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Extract random or clustered points from a PLY file and save as red points")
    parser.add_argument(
        "--input", 
        type=str, 
        help="Input PLY file path. If not provided, will use mouse.ply from demo/static/models/"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        help="Output PLY file path. If not provided, will use 'random_points.ply'"
    )
    parser.add_argument(
        "--num-points", 
        type=int, 
        default=3, 
        help="Number of random points to extract (default: 3)"
    )
    parser.add_argument(
        "--list-models", 
        action="store_true", 
        help="List available models in demo/static/models/"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        help="Choose a model from demo/static/models/ (e.g., 'mouse', 'rhino', 'scene')"
    )
    parser.add_argument(
        "--clustered", 
        action="store_true", 
        help="Select points that are close to each other instead of completely random"
    )
    parser.add_argument(
        "--cluster-radius", 
        type=float, 
        help="Maximum distance between points in a cluster (auto-estimated if not provided)"
    )
    
    args = parser.parse_args()
    
    # Models directory
    models_dir = "demo/static/models"
    
    # List available models if requested
    if args.list_models:
        print("Available models in demo/static/models/:")
        if os.path.exists(models_dir):
            ply_files = [f for f in os.listdir(models_dir) if f.endswith('.ply')]
            for i, ply_file in enumerate(sorted(ply_files), 1):
                print(f"  {i}. {ply_file}")
        else:
            print(f"  Models directory not found: {models_dir}")
        return
    
    # Determine input file
    if args.input:
        input_path = args.input
    elif args.model:
        # Use specified model from demo/static/models/
        if not args.model.endswith('.ply'):
            args.model += '.ply'
        input_path = os.path.join(models_dir, args.model)
    else:
        # Default to mouse.ply
        input_path = os.path.join(models_dir, "mouse.ply")
    
    # Determine output file
    if args.output:
        output_path = args.output
    else:
        # Generate output filename based on input and options
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        mode = "clustered" if args.clustered else "random"
        output_path = f"{base_name}_{mode}_{args.num_points}_points.ply"
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file does not exist: {input_path}")
        print("Use --list-models to see available models or provide a valid --input path")
        return
    
    # Set random seed for reproducibility (optional)
    random.seed(42)
    np.random.seed(42)
    
    # Extract random points
    try:
        extract_random_points(input_path, output_path, args.num_points, args.clustered, args.cluster_radius)
        mode = "clustered" if args.clustered else "random"
        print(f"\nSuccess! Created {output_path} with {args.num_points} {mode} red points.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

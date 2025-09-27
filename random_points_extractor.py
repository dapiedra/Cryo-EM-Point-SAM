#!/usr/bin/env python3
"""
Script to extract random points from a ground truth PLY file (A) and find their closest 
corresponding points in a target PLY file (B). This is useful when you have ground truth 
segmentation data in one file but need to perform segmentation on a different (but similar) file.

The script:
1. Loads the ground truth PLY file (A) containing the area to segment
2. Loads the target PLY file (B) that will be used for actual segmentation  
3. Extracts random/clustered points from the ground truth file
4. Finds the closest corresponding points in the target file
5. Saves the closest target points as red points for use as segmentation prompts
"""

import os
import random
import argparse
import numpy as np
from pc_sam.ply_utils import load_ply, save_ply


def find_closest_points(source_points, target_points):
    """
    Find the closest points in target_points for each point in source_points.
    
    Args:
        source_points (np.array): Array of source 3D points with shape [N, 3]
        target_points (np.array): Array of target 3D points with shape [M, 3]
    
    Returns:
        np.array: Closest points from target_points with shape [N, 3]
        np.array: Indices of closest points in target_points
        np.array: Distances to closest points
    """
    print(f"Finding closest points for {len(source_points)} points in target set of {len(target_points)} points...")
    
    closest_points = []
    closest_indices = []
    closest_distances = []
    
    for i, source_point in enumerate(source_points):
        # Calculate distances from this source point to all target points
        distances = np.sqrt(np.sum((target_points - source_point) ** 2, axis=1))
        
        # Find the index of the closest point
        closest_idx = np.argmin(distances)
        closest_distance = distances[closest_idx]
        closest_point = target_points[closest_idx]
        
        closest_points.append(closest_point)
        closest_indices.append(closest_idx)
        closest_distances.append(closest_distance)
        
        print(f"  Point {i+1}: Source ({source_point[0]:.6f}, {source_point[1]:.6f}, {source_point[2]:.6f}) "
              f"-> Target ({closest_point[0]:.6f}, {closest_point[1]:.6f}, {closest_point[2]:.6f}) "
              f"[distance: {closest_distance:.6f}]")
    
    return np.array(closest_points), np.array(closest_indices), np.array(closest_distances)


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


def extract_random_points(ground_truth_ply_path, target_ply_path, output_ply_path, num_points=3, clustered=False, cluster_radius=None):
    """
    Extract random points from a ground truth PLY file and find their closest matches in a target PLY file.
    Save the closest points from the target file as red points.
    
    Args:
        ground_truth_ply_path (str): Path to the ground truth PLY file (file A)
        target_ply_path (str): Path to the target PLY file (file B) for segmentation
        output_ply_path (str): Path to save the output PLY file
        num_points (int): Number of random points to extract (default: 3)
        clustered (bool): If True, select points close to each other
        cluster_radius (float): Maximum distance for clustered points (auto-estimated if None)
    """
    print(f"Loading ground truth PLY file: {ground_truth_ply_path}")
    
    # Load the ground truth PLY file (file A)
    ground_truth_data = load_ply(ground_truth_ply_path)
    print(f"Loaded {ground_truth_data.shape[0]} points from ground truth file")
    
    print(f"Loading target PLY file: {target_ply_path}")
    
    # Load the target PLY file (file B)
    target_data = load_ply(target_ply_path)
    print(f"Loaded {target_data.shape[0]} points from target file")
    
    # Extract only the xyz coordinates (first 3 columns) from ground truth
    ground_truth_xyz = ground_truth_data[:, :3]
    
    # Extract only the xyz coordinates (first 3 columns) from target
    target_xyz = target_data[:, :3]
    
    # Select points from ground truth based on the clustering option
    if clustered:
        print(f"Selecting {num_points} clustered points from ground truth...")
        selected_gt_points, selected_gt_indices = select_clustered_points(ground_truth_xyz, num_points, cluster_radius)
    else:
        print(f"Selecting {num_points} random points from ground truth...")
        # Randomly select the specified number of points from ground truth
        if ground_truth_data.shape[0] < num_points:
            print(f"Warning: Ground truth PLY file only has {ground_truth_data.shape[0]} points, selecting all of them.")
            selected_gt_indices = list(range(ground_truth_data.shape[0]))
        else:
            selected_gt_indices = random.sample(range(ground_truth_data.shape[0]), num_points)
        
        selected_gt_points = ground_truth_xyz[selected_gt_indices]
    
    print(f"Selected {len(selected_gt_points)} points from ground truth:")
    for i, point in enumerate(selected_gt_points):
        print(f"  GT Point {i+1}: ({point[0]:.6f}, {point[1]:.6f}, {point[2]:.6f})")
    
    # Find the closest points in the target file
    print(f"\nFinding closest matches in target file...")
    closest_target_points, closest_indices, distances = find_closest_points(selected_gt_points, target_xyz)
    
    print(f"\nMatching results:")
    print(f"  Average distance to closest matches: {np.mean(distances):.6f}")
    print(f"  Maximum distance to closest match: {np.max(distances):.6f}")
    print(f"  Minimum distance to closest match: {np.min(distances):.6f}")
    
    # Create red color for all selected points (RGB values in range [0, 1])
    red_colors = np.array([[1.0, 0.0, 0.0]] * len(closest_target_points))
    
    # Save the closest target points as red points
    save_ply(output_ply_path, closest_target_points, red_colors)
    
    print(f"\nSaved {len(closest_target_points)} red points (from target file) to: {output_ply_path}")
    
    return closest_target_points, closest_indices, distances


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Extract random or clustered points from a ground truth PLY file and find closest matches in a target PLY file")
    parser.add_argument(
        "--ground-truth", 
        type=str, 
        required=True,
        help="Ground truth PLY file path (file A) - points will be selected from this file"
    )
    parser.add_argument(
        "--target", 
        type=str, 
        required=True,
        help="Target PLY file path (file B) - closest matches will be found in this file for segmentation"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        help="Output PLY file path. If not provided, will use 'matched_points.ply'"
    )
    parser.add_argument(
        "--num-points", 
        type=int, 
        default=3, 
        help="Number of random points to extract (default: 3)"
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
    
    # Get file paths
    ground_truth_path = args.ground_truth
    target_path = args.target
    
    # Determine output file
    if args.output:
        output_path = args.output
    else:
        # Generate output filename based on input files and options
        gt_base_name = os.path.splitext(os.path.basename(ground_truth_path))[0]
        target_base_name = os.path.splitext(os.path.basename(target_path))[0]
        mode = "clustered" if args.clustered else "random"
        output_path = f"{gt_base_name}_to_{target_base_name}_{mode}_{args.num_points}_points.ply"
    
    # Check if input files exist
    if not os.path.exists(ground_truth_path):
        print(f"Error: Ground truth file does not exist: {ground_truth_path}")
        return
    
    if not os.path.exists(target_path):
        print(f"Error: Target file does not exist: {target_path}")
        return
    
    print(f"Ground truth file: {ground_truth_path}")
    print(f"Target file: {target_path}")
    print(f"Output file: {output_path}")
    
    # Set random seed for reproducibility (optional)
    random.seed(42)
    np.random.seed(42)
    
    # Extract random points and find closest matches
    try:
        closest_points, closest_indices, distances = extract_random_points(
            ground_truth_path, target_path, output_path, 
            args.num_points, args.clustered, args.cluster_radius
        )
        
        mode = "clustered" if args.clustered else "random"
        print(f"\nSuccess! Created {output_path} with {args.num_points} {mode} points matched from ground truth to target.")
        print(f"You can now use these points as prompts for segmentation on the target file: {target_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

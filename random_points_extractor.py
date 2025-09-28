#!/usr/bin/env python3
"""
Script to extract positive and negative points for Point-SAM segmentation from PLY files.
This is useful when you have ground truth segmentation data in one file but need to perform 
segmentation on a different (but similar) file.

The script:
1. Loads the ground truth PLY file (A) containing the area to segment
2. Loads the target PLY file (B) that will be used for actual segmentation  
3. Extracts positive points (random/clustered) from the ground truth file
4. Finds the closest corresponding points in the target file for positive prompts
5. Optionally generates negative points from areas in the target file that are far from ground truth
6. Saves points with appropriate colors: red for positive prompts, green for negative prompts
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


def farthest_point_sampling(points, num_samples):
    """
    Select points using farthest point sampling for better spatial distribution.
    
    Args:
        points (np.array): Array of 3D points with shape [N, 3]
        num_samples (int): Number of points to sample
    
    Returns:
        list: Indices of selected points
    """
    if len(points) <= num_samples:
        return list(range(len(points)))
    
    # Start with a random point
    selected_indices = [np.random.randint(0, len(points))]
    remaining_indices = list(range(len(points)))
    remaining_indices.remove(selected_indices[0])
    
    for _ in range(num_samples - 1):
        max_min_distance = -1
        farthest_idx = -1
        
        # Find the point that is farthest from all selected points
        for idx in remaining_indices:
            # Calculate minimum distance to any selected point
            min_dist_to_selected = float('inf')
            for selected_idx in selected_indices:
                dist = np.sqrt(np.sum((points[idx] - points[selected_idx]) ** 2))
                min_dist_to_selected = min(min_dist_to_selected, dist)
            
            # Keep track of the point with maximum minimum distance
            if min_dist_to_selected > max_min_distance:
                max_min_distance = min_dist_to_selected
                farthest_idx = idx
        
        if farthest_idx != -1:
            selected_indices.append(farthest_idx)
            remaining_indices.remove(farthest_idx)
    
    return selected_indices


def select_negative_points(target_points, ground_truth_points, num_negative_points=3, exclusion_radius=None, distributed=True):
    """
    Select random points from target file that are far from all ground truth points.
    
    Args:
        target_points (np.array): Array of target 3D points with shape [M, 3]
        ground_truth_points (np.array): Array of ground truth 3D points with shape [N, 3]
        num_negative_points (int): Number of negative points to select
        exclusion_radius (float): Minimum distance from ground truth points.
                                 If None, will be estimated automatically.
        distributed (bool): If True, use farthest point sampling for better distribution
    
    Returns:
        np.array: Selected negative points from target_points
        list: Indices of selected negative points in target_points
    """
    # Memory usage warning for large point clouds
    total_combinations = len(target_points) * len(ground_truth_points)
    if total_combinations > 10_000_000:  # 10 million combinations
        estimated_ram = (total_combinations * 8) / (1024**3)  # Estimate RAM in GB
        print(f"⚠️  Large point cloud detected:")
        print(f"   Target: {len(target_points):,} points, Ground truth: {len(ground_truth_points):,} points")
        print(f"   Using memory-efficient chunked processing to avoid RAM issues")
        print(f"   This may take a few minutes...")
    
    if len(target_points) < num_negative_points:
        print(f"Warning: Target file only has {len(target_points)} points, selecting all as potential negatives.")
        num_negative_points = len(target_points)
    
    # If exclusion_radius is not specified, estimate it based on ground truth coverage
    if exclusion_radius is None:
        # Calculate average distance between ground truth points to estimate coverage
        if len(ground_truth_points) > 1:
            # Use smaller sample for large point clouds to avoid memory issues
            sample_size = min(50, len(ground_truth_points))  # Reduced from 100 to 50
            sample_indices = np.random.choice(len(ground_truth_points), sample_size, replace=False)
            sample_gt = ground_truth_points[sample_indices]
            
            # More memory-efficient nearest neighbor distance calculation
            min_distances = []
            for i, point in enumerate(sample_gt):
                # Calculate distances to all other sample points
                other_points = np.delete(sample_gt, i, axis=0)
                if len(other_points) > 0:
                    distances = np.sqrt(np.sum((other_points - point) ** 2, axis=1))
                    min_distances.append(np.min(distances))
            
            if len(min_distances) > 0:
                avg_min_dist = np.mean(min_distances)
                # Use smaller multiplier for better point distribution - was 5x, now 2x
                exclusion_radius = avg_min_dist * 2  
                print(f"Auto-estimated exclusion radius from {sample_size} sample points: {exclusion_radius:.6f}")
                print(f"Using smaller exclusion radius for better negative point distribution")
            else:
                exclusion_radius = 1.0  # Increased fallback value for large point clouds
        else:
            exclusion_radius = 1.0  # Single point case
    
    print(f"Using exclusion radius: {exclusion_radius:.6f}")
    
    # For each target point, calculate minimum distance to any ground truth point
    print(f"Calculating distances from {len(target_points)} target points to {len(ground_truth_points)} ground truth points...")
    
    # Memory-efficient chunked distance calculation to avoid RAM issues with large point clouds
    chunk_size = min(1000, len(target_points))  # Process in chunks of 1000 points
    min_distances_to_gt = np.zeros(len(target_points))
    
    print(f"Processing in chunks of {chunk_size} points to avoid memory issues...")
    
    for start_idx in range(0, len(target_points), chunk_size):
        end_idx = min(start_idx + chunk_size, len(target_points))
        chunk_target = target_points[start_idx:end_idx]
        
        # Calculate distances for this chunk
        # Reshape for broadcasting: chunk_target [chunk_size, 1, 3] - ground_truth_points [1, N, 3]
        target_expanded = chunk_target[:, np.newaxis, :]  # [chunk_size, 1, 3]
        gt_expanded = ground_truth_points[np.newaxis, :, :]  # [1, N, 3]
        
        # Calculate distances for this chunk
        chunk_distances = np.sqrt(np.sum((target_expanded - gt_expanded) ** 2, axis=2))  # [chunk_size, N]
        
        # Find minimum distance for each point in this chunk
        min_distances_to_gt[start_idx:end_idx] = np.min(chunk_distances, axis=1)
        
        # Progress indicator
        if len(target_points) > 5000:  # Only show progress for large point clouds
            progress = (end_idx / len(target_points)) * 100
            print(f"  Progress: {progress:.1f}% ({end_idx}/{len(target_points)} points processed)")
    
    print(f"Distance calculation completed.")
    
    # Find candidate points that are far enough from all ground truth points
    candidate_indices = np.where(min_distances_to_gt >= exclusion_radius)[0]
    
    print(f"Found {len(candidate_indices)} candidate negative points (out of {len(target_points)} total target points)")
    
    if len(candidate_indices) == 0:
        print(f"Warning: No points found beyond exclusion radius {exclusion_radius:.6f}")
        print("Reducing exclusion radius and trying again...")
        exclusion_radius *= 0.5
        candidate_indices = np.where(min_distances_to_gt >= exclusion_radius)[0]
        print(f"With reduced exclusion radius {exclusion_radius:.6f}, found {len(candidate_indices)} candidates")
    
    if len(candidate_indices) == 0:
        print("Warning: Still no candidates found. Using points with maximum distance from ground truth...")
        # Take the points that are farthest from ground truth
        sorted_indices = np.argsort(min_distances_to_gt)[::-1]  # Sort by distance, descending
        candidate_indices = sorted_indices[:min(num_negative_points * 3, len(sorted_indices))]
    
    # Select points with better spatial distribution
    if len(candidate_indices) >= num_negative_points:
        if distributed and len(candidate_indices) > num_negative_points * 2:
            # Use farthest point sampling for better distribution
            print(f"Using farthest point sampling for better spatial distribution...")
            selected_indices = farthest_point_sampling(target_points[candidate_indices], num_negative_points)
            selected_indices = candidate_indices[selected_indices]  # Map back to original indices
        else:
            # Completely random selection from all valid candidates
            selected_indices = np.random.choice(candidate_indices, num_negative_points, replace=False)
    else:
        print(f"Warning: Only {len(candidate_indices)} candidates available, selecting all of them.")
        selected_indices = candidate_indices
    
    selected_negative_points = target_points[selected_indices]
    
    # Print statistics
    selected_distances = min_distances_to_gt[selected_indices]
    print(f"Negative point statistics:")
    print(f"  Number of negative points: {len(selected_negative_points)}")
    print(f"  Average distance to nearest ground truth: {np.mean(selected_distances):.6f}")
    print(f"  Min distance to nearest ground truth: {np.min(selected_distances):.6f}")
    print(f"  Max distance to nearest ground truth: {np.max(selected_distances):.6f}")
    
    return selected_negative_points, selected_indices.tolist()


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


def extract_random_points(ground_truth_ply_path, target_ply_path, output_ply_path, num_points=3, clustered=False, cluster_radius=None, num_negative_points=0, exclusion_radius=None, distributed=True):
    """
    Extract random points from a ground truth PLY file and find their closest matches in a target PLY file.
    Optionally generate negative points that are far from the ground truth area.
    Save the points with appropriate colors (red for positive, green for negative).
    
    Args:
        ground_truth_ply_path (str): Path to the ground truth PLY file (file A)
        target_ply_path (str): Path to the target PLY file (file B) for segmentation
        output_ply_path (str): Path to save the output PLY file
        num_points (int): Number of positive points to extract (default: 3)
        clustered (bool): If True, select positive points close to each other
        cluster_radius (float): Maximum distance for clustered positive points (auto-estimated if None)
        num_negative_points (int): Number of negative points to generate (default: 0)
        exclusion_radius (float): Minimum distance from ground truth for negative points (auto-estimated if None)
        distributed (bool): Use farthest point sampling for better negative point distribution
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
    
    # Start collecting all points and colors for output
    all_output_points = []
    all_output_colors = []
    
    # Handle positive points only if requested
    closest_target_points = None
    closest_indices = None
    distances = None
    
    if num_points > 0:
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
        
        # Add positive points (red color)
        red_colors = np.array([[1.0, 0.0, 0.0]] * len(closest_target_points))
        all_output_points.append(closest_target_points)
        all_output_colors.append(red_colors)
    else:
        print("No positive points requested (num_points=0)")
        closest_target_points = np.array([]).reshape(0, 3)  # Empty array for consistency
        closest_indices = np.array([])
        distances = np.array([])
    
    # Generate negative points if requested
    negative_points = None
    negative_indices = None
    if num_negative_points > 0:
        print(f"\nGenerating {num_negative_points} negative points...")
        negative_points, negative_indices = select_negative_points(
            target_xyz, ground_truth_xyz, num_negative_points, exclusion_radius, distributed=distributed
        )
        
        # Add negative points (green color)
        green_colors = np.array([[0.0, 1.0, 0.0]] * len(negative_points))
        all_output_points.append(negative_points)
        all_output_colors.append(green_colors)
        
        print(f"Selected {len(negative_points)} negative points from target file:")
        for i, point in enumerate(negative_points):
            print(f"  Negative Point {i+1}: ({point[0]:.6f}, {point[1]:.6f}, {point[2]:.6f})")
    
    # Combine all points and colors
    if all_output_points:
        final_points = np.vstack(all_output_points)
        final_colors = np.vstack(all_output_colors)
        
        # Save all points with appropriate colors
        save_ply(output_ply_path, final_points, final_colors)
        
        total_positive = len(closest_target_points)
        total_negative = len(negative_points) if negative_points is not None else 0
        print(f"\nSaved {total_positive + total_negative} points to: {output_ply_path}")
        print(f"  - {total_positive} positive points (red)")
        print(f"  - {total_negative} negative points (green)")
    
    return closest_target_points, closest_indices, distances, negative_points, negative_indices


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Extract positive and negative points for Point-SAM segmentation from PLY files")
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
    parser.add_argument(
        "--negative-points", 
        type=int, 
        default=0, 
        help="Number of negative points to generate (default: 0, meaning no negative points)"
    )
    parser.add_argument(
        "--exclusion-radius", 
        type=float, 
        help="Minimum distance from ground truth for negative points (auto-estimated if not provided)"
    )
    parser.add_argument(
        "--distributed", 
        action="store_true", 
        default=True,
        help="Use farthest point sampling for better negative point distribution (default: True)"
    )
    
    args = parser.parse_args()
    
    # Validate that at least one type of point is requested
    if args.num_points == 0 and args.negative_points == 0:
        print("Error: Must request at least one positive point (--num-points > 0) or negative point (--negative-points > 0)")
        return
    
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
        
        # Generate filename based on what points are requested
        if args.num_points > 0 and args.negative_points > 0:
            # Both positive and negative points
            output_path = f"{gt_base_name}_to_{target_base_name}_{mode}_{args.num_points}pos_{args.negative_points}neg_points.ply"
        elif args.num_points > 0:
            # Only positive points
            output_path = f"{gt_base_name}_to_{target_base_name}_{mode}_{args.num_points}_points.ply"
        else:
            # Only negative points
            output_path = f"{gt_base_name}_to_{target_base_name}_negative_{args.negative_points}_points.ply"
    
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
        closest_points, closest_indices, distances, negative_points, negative_indices = extract_random_points(
            ground_truth_path, target_path, output_path, 
            args.num_points, args.clustered, args.cluster_radius,
            args.negative_points, args.exclusion_radius, args.distributed
        )
        
        mode = "clustered" if args.clustered else "random"
        total_positive = args.num_points
        total_negative = args.negative_points
        
        print(f"\nSuccess! Created {output_path} with:")
        print(f"  - {total_positive} {mode} positive points (red) matched from ground truth to target")
        if total_negative > 0:
            print(f"  - {total_negative} negative points (green) from areas outside ground truth")
        print(f"You can now use these points as prompts for segmentation on the target file: {target_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

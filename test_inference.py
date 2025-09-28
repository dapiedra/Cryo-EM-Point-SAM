#!/usr/bin/env python3
"""
Test script to understand Point-SAM inference without the graphical interface.
This script demonstrates how to use Point-SAM with your own .ply files.
"""

import sys
import os
import argparse
import time
import hydra
from omegaconf import OmegaConf
import numpy as np
import torch
from pc_sam.model.pc_sam import PointCloudSAM
from pc_sam.utils.torch_utils import replace_with_fused_layernorm
from safetensors.torch import load_model
from pc_sam.ply_utils import load_ply, save_ply, visualize_mask

# Global configuration for prompt point limits
MAX_POSITIVE_POINTS = 1  # Maximum number of positive (red) prompt points to use
MAX_NEGATIVE_POINTS = 100000  # Maximum number of negative (green) prompt points to use

# Random seed configuration
# To use a fixed seed for reproducible results, replace the line below with: RANDOM_SEED = 42
RANDOM_SEED = int(time.time() * 1000000) % 2147483647  # Generate random seed based on current time


def normalize_points(points):
    """Normalize point cloud to unit sphere."""
    shift = points.mean(0)
    scale = np.linalg.norm(points - shift, axis=-1).max()
    return (points - shift) / scale


def load_prompt_ply(filepath):
    """
    Load prompt points from a .ply file.
    Expected format: .ply file where RGB colors encode the labels:
    - Red points (R>200, G<100, B<100) = positive prompts (label=1)
    - Green points (R<100, G>200, B<100) = negative prompts (label=0)
    - Other colors are filtered out (ignored)
    
    Returns:
        prompt_coords: numpy array of shape [N, 3] with 3D coordinates
        prompt_labels: numpy array of shape [N,] with labels (0 or 1)
    """
    try:
        # Set random seed for reproducible point selection
        print(f"Using random seed: {RANDOM_SEED}")
        np.random.seed(RANDOM_SEED)
        
        points = load_ply(filepath)
        coords = points[:, :3]  # XYZ coordinates
        rgb = points[:, 3:6]    # RGB colors (0-255 range)
        
        # Filter and determine labels based on RGB colors
        positive_coords = []
        negative_coords = []
        positive_rgb = []
        negative_rgb = []
        
        for i in range(len(rgb)):
            r, g, b = rgb[i]
            # Check if it's green (negative prompt)
            if g > 200 and r < 100 and b < 100:  # Mostly green
                negative_coords.append(coords[i])
                negative_rgb.append(rgb[i])
            # Check if it's red (positive prompt)
            elif r > 200 and g < 100 and b < 100:  # Mostly red
                positive_coords.append(coords[i])
                positive_rgb.append(rgb[i])
            # Skip all other colors (not red or green enough)
        
        # Limit the number of positive and negative points
        if len(positive_coords) > MAX_POSITIVE_POINTS:
            # Randomly select MAX_POSITIVE_POINTS from available positive points
            indices = np.random.choice(len(positive_coords), MAX_POSITIVE_POINTS, replace=False)
            positive_coords = [positive_coords[i] for i in indices]
            positive_rgb = [positive_rgb[i] for i in indices]
            
        if len(negative_coords) > MAX_NEGATIVE_POINTS:
            # Randomly select MAX_NEGATIVE_POINTS from available negative points
            indices = np.random.choice(len(negative_coords), MAX_NEGATIVE_POINTS, replace=False)
            negative_coords = [negative_coords[i] for i in indices]
            negative_rgb = [negative_rgb[i] for i in indices]
        
        # Combine positive and negative points
        filtered_coords = positive_coords + negative_coords
        filtered_rgb = positive_rgb + negative_rgb
        labels = [1] * len(positive_coords) + [0] * len(negative_coords)
        
        if len(filtered_coords) == 0:
            raise ValueError("No valid red or green prompt points found in the PLY file")
        
        coords = np.array(filtered_coords)
        labels = np.array(labels)
        rgb = np.array(filtered_rgb)
        
        original_points = len(points)
        # Count original red/green points before limiting
        original_positive = len([1 for i in range(len(points)) if points[i][5] < 100 and points[i][4] < 100 and points[i][3] > 200])
        original_negative = len([1 for i in range(len(points)) if points[i][3] < 100 and points[i][5] < 100 and points[i][4] > 200])
        
        final_points = len(coords)
        
        print(f"Original points in PLY: {original_points}")
        print(f"Point filtering and limiting applied:")
        print(f"  - Max positive points allowed: {MAX_POSITIVE_POINTS}")
        print(f"  - Max negative points allowed: {MAX_NEGATIVE_POINTS}")
        print(f"  - Red points found: {original_positive}")
        print(f"  - Green points found: {original_negative}")
        print(f"Final prompt points: {final_points}")
        print(f"  - Positive prompts used (red, label=1): {np.sum(labels == 1)}")
        print(f"  - Negative prompts used (green, label=0): {np.sum(labels == 0)}")
        
        if original_positive > MAX_POSITIVE_POINTS:
            print(f"  - Limited positive points: {original_positive} -> {np.sum(labels == 1)} (randomly selected)")
        if original_negative > MAX_NEGATIVE_POINTS:
            print(f"  - Limited negative points: {original_negative} -> {np.sum(labels == 0)} (randomly selected)")
        
        # Print detailed coordinates for each final point
        print("\nFinal prompt points (after filtering and limiting):")
        for i, (coord, label) in enumerate(zip(coords, labels)):
            point_type = "POSITIVE (RED)" if label == 1 else "NEGATIVE (GREEN)"
            rgb_info = rgb[i]
            print(f"  Point {i+1}: {point_type} - Coordinates: ({coord[0]:.6f}, {coord[1]:.6f}, {coord[2]:.6f}) - RGB: ({rgb_info[0]:.0f}, {rgb_info[1]:.0f}, {rgb_info[2]:.0f})")
        
        return coords, labels
        
    except Exception as e:
        raise ValueError(f"Error loading prompt .ply file: {e}")


def main():
    parser = argparse.ArgumentParser(description="Point-SAM inference without GUI")
    parser.add_argument("--config", type=str, default="large", help="Config name")
    parser.add_argument("--config_dir", type=str, default="./configs", help="Config directory")
    parser.add_argument("--ckpt_path", type=str, 
                       default="./pretrained/model.safetensors", 
                       help="Checkpoint path")
    parser.add_argument("--pointcloud", type=str, required=True, 
                       help="Path to input .ply file")
    
    # Modified to accept either manual coordinates or .ply file for prompts
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt_points", type=str,
                             help="Comma-separated coordinates of prompt points (x1,y1,z1;x2,y2,z2)")
    prompt_group.add_argument("--prompt_ply", type=str,
                             help="Path to .ply file with prompt points (red=positive, green=negative)")
    
    parser.add_argument("--prompt_labels", type=str, default="1",
                       help="Comma-separated labels for prompts when using --prompt_points (1=positive, 0=negative)")
    parser.add_argument("--output", type=str, default="output_mask.ply",
                       help="Output .ply file path")
    
    args = parser.parse_args()
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    
    print(f"Loading Point-SAM model...")
    
    # Load configuration
    with hydra.initialize(args.config_dir, version_base=None):
        cfg = hydra.compose(config_name=args.config)
        OmegaConf.resolve(cfg)
    
    # Setup model
    model: PointCloudSAM = hydra.utils.instantiate(cfg.model)
    model.apply(replace_with_fused_layernorm)
    
    # Load pre-trained model
    load_model(model, args.ckpt_path)
    model.eval()
    model.cuda()
    
    print(f"Loading point cloud from {args.pointcloud}...")
    
    # Load point cloud
    points = load_ply(args.pointcloud)
    xyz_original = points[:, :3]  # Keep original coordinates
    xyz = points[:, :3]
    rgb = points[:, 3:6] / 255.0  # Normalize RGB to [0,1]
    
    # Normalize coordinates using the same method as the demo
    shift = xyz.mean(0)
    scale = np.linalg.norm(xyz - shift, axis=-1).max()
    xyz = (xyz - shift) / scale
    
    print(f"Point cloud shape: {xyz.shape}")
    print(f"Normalization - shift: {shift}, scale: {scale}")
    
    # Convert to tensors
    pc_xyz = torch.from_numpy(xyz).cuda().float().unsqueeze(0)  # [1, N, 3]
    pc_rgb = torch.from_numpy(rgb).cuda().float().unsqueeze(0)  # [1, N, 3]
    
    # Load prompt points - either from .ply file or manual coordinates
    if args.prompt_ply:
        print(f"Loading prompt points from {args.prompt_ply}...")
        prompt_coords_raw, prompt_labels_list = load_prompt_ply(args.prompt_ply)
        
        # Apply the same normalization as the point cloud to prompt points
        prompt_coords_list = []
        print(f"\nNormalizing prompt points using shift: {shift}, scale: {scale}")
        print("Normalized prompt points:")
        for i, coords in enumerate(prompt_coords_raw):
            normalized_coords = (coords - shift) / scale
            prompt_coords_list.append(normalized_coords.tolist())
            point_type = "POSITIVE" if prompt_labels_list[i] == 1 else "NEGATIVE"
            print(f"  Point {i+1}: {point_type} - Original: ({coords[0]:.6f}, {coords[1]:.6f}, {coords[2]:.6f}) -> Normalized: ({normalized_coords[0]:.6f}, {normalized_coords[1]:.6f}, {normalized_coords[2]:.6f})")
            
    else:
        # Parse prompt points from command line and apply normalization
        prompt_coords_list = []
        print(f"\nNormalizing manual prompt points using shift: {shift}, scale: {scale}")
        print("Normalized manual prompt points:")
        for i, point_str in enumerate(args.prompt_points.split(';')):
            coords = [float(x) for x in point_str.split(',')]
            if len(coords) != 3:
                raise ValueError("Each prompt point must have exactly 3 coordinates (x,y,z)")
            # Apply the same normalization as the point cloud
            normalized_coords = (np.array(coords) - shift) / scale
            prompt_coords_list.append(normalized_coords.tolist())
        
        # Parse prompt labels
        prompt_labels_list = [int(x) for x in args.prompt_labels.split(',')]
        
        # Ensure we have labels for all prompt points
        if len(prompt_labels_list) == 1 and len(prompt_coords_list) > 1:
            prompt_labels_list = prompt_labels_list * len(prompt_coords_list)
        
        # Print detailed information
        for i, (coords_orig, coords_norm, label) in enumerate(zip([point_str.split(',') for point_str in args.prompt_points.split(';')], prompt_coords_list, prompt_labels_list)):
            point_type = "POSITIVE" if label == 1 else "NEGATIVE"
            orig_coords = [float(x) for x in coords_orig]
            print(f"  Point {i+1}: {point_type} - Original: ({orig_coords[0]:.6f}, {orig_coords[1]:.6f}, {orig_coords[2]:.6f}) -> Normalized: ({coords_norm[0]:.6f}, {coords_norm[1]:.6f}, {coords_norm[2]:.6f})")
    
    if len(prompt_coords_list) != len(prompt_labels_list):
        raise ValueError("Number of prompt points must match number of labels")
    
    prompt_coords = torch.from_numpy(np.array(prompt_coords_list)).cuda().float()[None, ...]  # [1, M, 3]
    prompt_labels = torch.from_numpy(np.array(prompt_labels_list)).cuda()[None, ...]  # [1, M]
    
    print(f"Prompt points: {prompt_coords_list}")
    print(f"Prompt labels: {prompt_labels_list}")
    
    print("Running inference...")
    
    # Run inference
    with torch.no_grad():
        masks, scores = model.predict_masks(
            pc_xyz, pc_rgb, prompt_coords, prompt_labels, 
            prompt_masks=None, multimask_output=True
        )
    
    # Get the best mask
    best_mask_idx = torch.argmax(scores[0])
    best_mask = masks[0][best_mask_idx] > 0  # [N,]
    
    print(f"Generated {masks.shape[1]} masks, best mask covers {best_mask.sum()} points")
    
    # Save result using original coordinates
    visualize_mask(args.output, xyz_original, best_mask.cpu().numpy())
    print(f"Saved segmentation result to {args.output}")
    print(f"Output coordinates are in original space (before normalization)")
    
    # Print some statistics
    total_points = len(xyz)
    segmented_points = best_mask.sum().item()
    percentage = (segmented_points / total_points) * 100
    print(f"Segmented {segmented_points}/{total_points} points ({percentage:.1f}%)")


if __name__ == "__main__":
    main()

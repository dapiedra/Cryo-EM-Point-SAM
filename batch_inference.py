#!/usr/bin/env python3
"""
Batch inference script for Point-SAM to process multiple point cloud pairs.
This script processes folders containing point cloud files (colored_*.ply) and their
corresponding prompt files (points_*.ply) and generates prediction files.
"""

import sys
import os
import glob
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
        
        return coords, labels
        
    except Exception as e:
        raise ValueError(f"Error loading prompt .ply file: {e}")


def find_file_pairs(input_folder):
    """
    Find pairs of colored_*.ply and points_*.ply files in the input folder.
    
    Returns:
        List of tuples: (colored_file_path, points_file_path, identifier)
        where identifier is the part that matches between files (e.g., "10274_1")
    """
    # Find all colored files
    colored_pattern = os.path.join(input_folder, "colored_*.ply")
    colored_files = glob.glob(colored_pattern)
    
    pairs = []
    
    for colored_file in colored_files:
        # Extract the identifier from colored file
        basename = os.path.basename(colored_file)
        if basename.startswith("colored_") and basename.endswith(".ply"):
            identifier = basename[8:-4]  # Remove "colored_" prefix and ".ply" suffix
            
            # Look for corresponding points file
            points_file = os.path.join(input_folder, f"points_{identifier}.ply")
            
            if os.path.exists(points_file):
                pairs.append((colored_file, points_file, identifier))
            else:
                print(f"Warning: No matching points file found for {basename} (looking for points_{identifier}.ply)")
    
    return pairs


def process_single_pair(model, colored_file, points_file, output_file):
    """
    Process a single pair of point cloud and prompt files.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"\nProcessing pair:")
        print(f"  Point cloud: {os.path.basename(colored_file)}")
        print(f"  Prompt file: {os.path.basename(points_file)}")
        print(f"  Output: {os.path.basename(output_file)}")
        
        # Load point cloud
        points = load_ply(colored_file)
        xyz_original = points[:, :3]  # Keep original coordinates
        xyz = points[:, :3]
        rgb = points[:, 3:6] / 255.0  # Normalize RGB to [0,1]
        
        # Normalize coordinates using the same method as the demo
        shift = xyz.mean(0)
        scale = np.linalg.norm(xyz - shift, axis=-1).max()
        xyz = (xyz - shift) / scale
        
        print(f"  Point cloud shape: {xyz.shape}")
        
        # Convert to tensors
        pc_xyz = torch.from_numpy(xyz).cuda().float().unsqueeze(0)  # [1, N, 3]
        pc_rgb = torch.from_numpy(rgb).cuda().float().unsqueeze(0)  # [1, N, 3]
        
        # Load prompt points
        prompt_coords_raw, prompt_labels_list = load_prompt_ply(points_file)
        
        # Apply the same normalization as the point cloud to prompt points
        prompt_coords_list = []
        for coords in prompt_coords_raw:
            normalized_coords = (coords - shift) / scale
            prompt_coords_list.append(normalized_coords.tolist())
        
        prompt_coords = torch.from_numpy(np.array(prompt_coords_list)).cuda().float()[None, ...]  # [1, M, 3]
        prompt_labels = torch.from_numpy(np.array(prompt_labels_list)).cuda()[None, ...]  # [1, M]
        
        print(f"  Prompt points: {len(prompt_coords_list)} ({np.sum(prompt_labels_list == 1)} positive, {np.sum(prompt_labels_list == 0)} negative)")
        
        # Run inference
        with torch.no_grad():
            masks, scores = model.predict_masks(
                pc_xyz, pc_rgb, prompt_coords, prompt_labels, 
                prompt_masks=None, multimask_output=True
            )
        
        # Get the best mask
        best_mask_idx = torch.argmax(scores[0])
        best_mask = masks[0][best_mask_idx] > 0  # [N,]
        
        # Save result using original coordinates
        visualize_mask(output_file, xyz_original, best_mask.cpu().numpy())
        
        # Print statistics
        total_points = len(xyz)
        segmented_points = best_mask.sum().item()
        percentage = (segmented_points / total_points) * 100
        print(f"  Result: {segmented_points}/{total_points} points ({percentage:.1f}%) segmented")
        
        return True
        
    except Exception as e:
        print(f"  Error processing pair: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Point-SAM batch inference")
    parser.add_argument("--config", type=str, default="large", help="Config name")
    parser.add_argument("--config_dir", type=str, default="./configs", help="Config directory")
    parser.add_argument("--ckpt_path", type=str, 
                       default="./pretrained/model.safetensors", 
                       help="Checkpoint path")
    parser.add_argument("--input_folder", type=str, required=True, 
                       help="Path to input folder containing colored_*.ply and points_*.ply files")
    parser.add_argument("--output_folder", type=str, required=True,
                       help="Path to output folder for prediction_*.ply files")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing output files")
    
    args = parser.parse_args()
    
    # Validate input folder
    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder does not exist: {args.input_folder}")
        sys.exit(1)
    
    # Create output folder if needed
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder, exist_ok=True)
        print(f"Created output folder: {args.output_folder}")
    
    # Find file pairs
    print(f"Scanning input folder: {args.input_folder}")
    pairs = find_file_pairs(args.input_folder)
    
    if not pairs:
        print("No valid file pairs found in input folder!")
        print("Expected format:")
        print("  - colored_XXXX.ply (point cloud files)")
        print("  - points_XXXX.ply (prompt files)")
        print("Where XXXX is the same identifier for matching pairs")
        sys.exit(1)
    
    print(f"Found {len(pairs)} file pairs to process:")
    for colored_file, points_file, identifier in pairs:
        output_file = os.path.join(args.output_folder, f"prediction_{identifier}.ply")
        print(f"  {identifier}: {os.path.basename(colored_file)} + {os.path.basename(points_file)} -> {os.path.basename(output_file)}")
        
        # Check if output file already exists
        if os.path.exists(output_file) and not args.overwrite:
            print(f"    (Output file exists, will skip unless --overwrite is used)")
    
    print(f"\nLoading Point-SAM model...")
    print(f"Using random seed: {RANDOM_SEED}")
    
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
    
    print("Model loaded successfully!")
    
    # Process each pair
    successful = 0
    failed = 0
    skipped = 0
    
    for i, (colored_file, points_file, identifier) in enumerate(pairs):
        output_file = os.path.join(args.output_folder, f"prediction_{identifier}.ply")
        
        print(f"\n{'='*60}")
        print(f"Processing pair {i+1}/{len(pairs)}: {identifier}")
        
        # Check if output file already exists
        if os.path.exists(output_file) and not args.overwrite:
            print(f"Output file already exists, skipping: {output_file}")
            skipped += 1
            continue
        
        # Process the pair
        success = process_single_pair(model, colored_file, points_file, output_file)
        
        if success:
            successful += 1
            print(f"✓ Successfully saved: {output_file}")
        else:
            failed += 1
            print(f"✗ Failed to process pair: {identifier}")
    
    # Summary
    print(f"\n{'='*60}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total pairs found: {len(pairs)}")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped (already exists): {skipped}")
    print(f"Output folder: {args.output_folder}")
    
    if failed > 0:
        print(f"\nWarning: {failed} files failed to process. Check the error messages above.")
        sys.exit(1)
    else:
        print(f"\nAll files processed successfully!")


if __name__ == "__main__":
    main()
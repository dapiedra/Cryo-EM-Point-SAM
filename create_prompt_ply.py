#!/usr/bin/env python3
"""
Utility script to create prompt .ply files for Point-SAM.
This script helps you create prompt files with positive and negative points.
"""

import argparse
import numpy as np
from pc_sam.ply_utils import save_ply


def create_prompt_ply(output_path, positive_points=None, negative_points=None):
    """
    Create a .ply file with prompt points.
    
    Args:
        output_path: Path to save the .ply file
        positive_points: List of [x, y, z] coordinates for positive prompts
        negative_points: List of [x, y, z] coordinates for negative prompts
    """
    all_points = []
    all_colors = []
    
    # Add positive points (red color)
    if positive_points:
        for point in positive_points:
            all_points.append(point)
            all_colors.append([1.0, 0.0, 0.0])  # Red for positive
    
    # Add negative points (green color)
    if negative_points:
        for point in negative_points:
            all_points.append(point)
            all_colors.append([0.0, 1.0, 0.0])  # Green for negative
    
    if not all_points:
        raise ValueError("Must provide at least one positive or negative point")
    
    points = np.array(all_points)
    colors = np.array(all_colors)
    
    save_ply(output_path, points, colors)
    
    print(f"Created prompt .ply file: {output_path}")
    print(f"  - Positive points (red): {len(positive_points) if positive_points else 0}")
    print(f"  - Negative points (green): {len(negative_points) if negative_points else 0}")


def main():
    parser = argparse.ArgumentParser(description="Create prompt .ply files for Point-SAM")
    parser.add_argument("--output", type=str, required=True,
                       help="Output .ply file path")
    parser.add_argument("--positive", type=str,
                       help="Positive prompt points (x1,y1,z1;x2,y2,z2)")
    parser.add_argument("--negative", type=str,
                       help="Negative prompt points (x1,y1,z1;x2,y2,z2)")
    
    args = parser.parse_args()
    
    # Parse positive points
    positive_points = []
    if args.positive:
        for point_str in args.positive.split(';'):
            coords = [float(x) for x in point_str.split(',')]
            if len(coords) != 3:
                raise ValueError("Each point must have exactly 3 coordinates (x,y,z)")
            positive_points.append(coords)
    
    # Parse negative points
    negative_points = []
    if args.negative:
        for point_str in args.negative.split(';'):
            coords = [float(x) for x in point_str.split(',')]
            if len(coords) != 3:
                raise ValueError("Each point must have exactly 3 coordinates (x,y,z)")
            negative_points.append(coords)
    
    if not positive_points and not negative_points:
        raise ValueError("Must provide at least one positive or negative point using --positive or --negative")
    
    create_prompt_ply(args.output, positive_points, negative_points)


if __name__ == "__main__":
    main()

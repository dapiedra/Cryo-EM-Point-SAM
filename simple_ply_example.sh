#!/bin/bash

# Simple Point-SAM PLY Runner
# This script demonstrates the easiest way to run Point-SAM with .ply files

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}=== Simple Point-SAM PLY Runner ===${NC}"
echo ""

# Check if we have the required files
if [ ! -f "test_inference.py" ]; then
    echo -e "${RED}Error: test_inference.py not found${NC}"
    exit 1
fi

if [ ! -f "create_prompt_ply.py" ]; then
    echo -e "${RED}Error: create_prompt_ply.py not found${NC}"
    exit 1
fi

# Create directories if they don't exist
mkdir -p input output

echo -e "${GREEN}Step 1: Create a prompt file${NC}"
echo "Creating prompt points (red=positive, green=negative)..."

# Example: Create prompt file with positive and negative points
python create_prompt_ply.py \
    --output input/example_prompts.ply \
    --positive "0.1,0.2,0.3;0.5,0.6,0.1" \
    --negative "1.0,1.0,1.0"

echo ""
echo -e "${GREEN}Step 2: Run Point-SAM segmentation${NC}"

# Check if we have example data
if [ -f "input/points_4400_1.ply" ]; then
    POINTCLOUD="input/points_4400_1.ply"
    echo "Using existing point cloud: $POINTCLOUD"
elif [ -f "demo/static/models/scene.ply" ]; then
    POINTCLOUD="demo/static/models/scene.ply"
    echo "Using demo point cloud: $POINTCLOUD"
else
    echo -e "${YELLOW}No example point cloud found.${NC}"
    echo "Please place a .ply file in the input/ directory and run:"
    echo ""
    echo -e "${BLUE}python test_inference.py \\"
    echo "    --pointcloud input/your_file.ply \\"
    echo "    --prompt_ply input/example_prompts.ply \\"
    echo "    --output output/segmented_result.ply${NC}"
    echo ""
    exit 0
fi

# Run the segmentation
echo "Running segmentation..."
python test_inference.py \
    --config large \
    --ckpt_path ./pretrained/model.safetensors \
    --pointcloud "$POINTCLOUD" \
    --prompt_ply input/example_prompts.ply \
    --output output/segmented_result.ply

echo ""
echo -e "${GREEN}=== Complete! ===${NC}"
echo -e "Input point cloud: ${BLUE}$POINTCLOUD${NC}"
echo -e "Prompt file: ${BLUE}input/example_prompts.ply${NC}"
echo -e "Output segmentation: ${BLUE}output/segmented_result.ply${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Open output/segmented_result.ply in a 3D viewer (like MeshLab)"
echo "2. Red points = segmented region, White points = background"
echo "3. Modify input/example_prompts.ply to adjust prompts"
echo "4. Re-run this script to try different segmentations"
echo ""
echo -e "${BLUE}Usage with your own files:${NC}"
echo "python test_inference.py \\"
echo "    --pointcloud input/your_pointcloud.ply \\"
echo "    --prompt_ply input/your_prompts.ply \\"
echo "    --output output/your_result.ply"
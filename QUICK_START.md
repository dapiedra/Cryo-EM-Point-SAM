# Point-SAM Quick Start Guide

This guide helps you get started with Point-SAM for 3D point cloud segmentation using .ply files.

## 🚀 Quick Start

### Prerequisites
1. **Download the pretrained model**:
   ```bash
   mkdir -p pretrained
   # Download from: https://huggingface.co/yuchen0187/Point-SAM/tree/main
   # Place model.safetensors in ./pretrained/
   ```

2. **Check your setup**:
   ```bash
   ./run_point_sam.sh setup
   ```

### Basic Usage

#### 1. Interactive Demo (Web Browser)
```bash
# Start the demo server
./run_point_sam.sh demo --pointcloud demo/static/models/scene.ply

# Then open http://localhost:5000 in your browser
# Click on points to create prompts, then segment!
```

#### 2. Command-Line Inference

**Method A: Manual coordinates**
```bash
./run_point_sam.sh inference \
    --pointcloud input/points.ply \
    --prompt_points "0.1,0.2,0.3;-0.1,0.5,0.2" \
    --prompt_labels "1,0" \
    --output output/result.ply
```

**Method B: Prompt .ply file**
```bash
# Create prompt file
./run_point_sam.sh create_prompts \
    --output prompts.ply \
    --positive "1.2,0.5,-0.8;2.1,1.0,0.3" \
    --negative "0.0,0.0,0.0"

# Run segmentation
./run_point_sam.sh inference \
    --pointcloud input/points.ply \
    --prompt_ply prompts.ply \
    --output output/result.ply
```

#### 3. Batch Processing
```bash
# Process all .ply files in input/ directory
# Looks for matching *_prompts.ply files
./run_point_sam.sh batch --input_dir input/ --output_dir output/
```

## 📂 File Organization

```
Point-SAM/
├── run_point_sam.sh           # Main runner script
├── test_inference.py          # Non-GUI inference script
├── create_prompt_ply.py       # Utility to create prompt files
├── random_points_extractor.py # Extract prompts from ground truth
├── input/                     # Place your .ply files here
│   ├── points_4400_1.ply
│   ├── gt_4400_1.ply
│   └── map_4400_1.ply
├── output/                    # Segmentation results go here
│   └── mask_4400_1.ply
├── pretrained/                # Download checkpoint here
│   └── model.safetensors
└── demo/                      # Interactive demo
    └── static/models/         # Example point clouds
```

## 🎯 Understanding Prompts

Point-SAM works with **point prompts** - you click on 3D points to indicate what you want to segment:

- **Positive prompts** (red): Points that belong to your target object
- **Negative prompts** (green): Points that do NOT belong to your target object

### Creating Prompt Files

#### Option 1: Manual coordinates
```bash
./run_point_sam.sh create_prompts \
    --output chair_prompts.ply \
    --positive "0.5,0.2,0.1;0.3,0.8,0.0" \
    --negative "1.0,1.0,1.0"
```

#### Option 2: Extract from ground truth
```bash
./run_point_sam.sh extract_points \
    --ground_truth gt_4400_1.ply \
    --target points_4400_1.ply \
    --output extracted_prompts.ply \
    --num_points 5 \
    --clustered
```

#### Option 3: Use the interactive demo
1. Run: `./run_point_sam.sh demo --pointcloud your_file.ply`
2. Click on points in the web interface
3. Export prompts using the "Export Prompts (.ply)" button

## 🔧 Advanced Usage

### Custom Model Configuration
```bash
./run_point_sam.sh inference \
    --config base \
    --checkpoint ./pretrained/custom_model.safetensors \
    --pointcloud input.ply \
    --prompt_ply prompts.ply
```

### Demo Server Options
```bash
./run_point_sam.sh demo \
    --pointcloud scene.ply \
    --host 0.0.0.0 \
    --port 8080
```

## 📊 Working with Your Data

### Coordinate Systems
- Point-SAM normalizes all coordinates to a unit sphere
- Prompt coordinates must be in the **same coordinate system** as your point cloud
- The scripts handle normalization automatically

### File Formats
- **Input**: .ply files with XYZ coordinates and RGB colors
- **Prompts**: .ply files with colored points (red=positive, green=negative)
- **Output**: .ply files with segmentation masks (red=segmented, white=background)

### Example Workflow
```bash
# 1. Check setup
./run_point_sam.sh setup

# 2. Create prompts for your object
./run_point_sam.sh create_prompts \
    --output table_prompts.ply \
    --positive "1.2,0.0,0.5;1.8,0.0,0.3" \
    --negative "0.0,2.0,0.0"

# 3. Run segmentation
./run_point_sam.sh inference \
    --pointcloud furniture_scene.ply \
    --prompt_ply table_prompts.ply \
    --output table_segmented.ply

# 4. View results in your favorite 3D viewer
```

## 🚦 Troubleshooting

### Common Issues

1. **"Checkpoint not found"**
   - Download model.safetensors from HuggingFace to ./pretrained/

2. **"CUDA not available"**
   - Inference requires a CUDA-capable GPU
   - Use the demo for CPU-only visualization

3. **"No prompt points provided"**
   - You must specify either --prompt_points or --prompt_ply

4. **"Point cloud normalization issues"**
   - Ensure prompt coordinates are in the same space as your point cloud
   - Use the interactive demo to get correct coordinates

### Getting Help
```bash
./run_point_sam.sh help
```

## 🎬 Example Sessions

### Segmenting a Chair
```bash
# Using the provided example files
./run_point_sam.sh inference \
    --pointcloud input/points_4400_1.ply \
    --prompt_points "0.1,0.2,0.3" \
    --output output/chair_segment.ply
```

### Interactive Exploration
```bash
# Start demo with example data
./run_point_sam.sh demo --pointcloud demo/static/models/scene.ply

# Open browser to http://localhost:5000
# Click on objects to create prompts
# See real-time segmentation
```

### Batch Processing Multiple Objects
```bash
# Organize files
mkdir input
cp scene1.ply input/
cp scene1_prompts.ply input/  # Red/green prompt points
cp scene2.ply input/
cp scene2_prompts.ply input/

# Process all
./run_point_sam.sh batch --input_dir input/ --output_dir results/
```

## 📚 Additional Resources

- **Original Paper**: [Point-SAM: Promptable 3D Segmentation Model for Point Clouds](https://arxiv.org/abs/2406.17741)
- **Project Page**: https://point-sam.github.io
- **HuggingFace Demo**: https://huggingface.co/spaces/yuchen0187/Point-SAM
- **Model Checkpoints**: https://huggingface.co/yuchen0187/Point-SAM

## 🔧 Advanced Topics

### Training Your Own Model
```bash
# Use the provided training script
bash scripts/train_large.sh
```

### Custom Data Processing
- See `pc_sam/ply_utils.py` for PLY file handling
- See `random_points_extractor.py` for custom prompt generation
- See `test_inference.py` for programmatic usage examples

### Integration with Other Tools
The Point-SAM infrastructure can be integrated into larger pipelines:
- Use `test_inference.py` as a Python module
- Call the runner script from other bash scripts
- Build custom GUIs using the model inference code

---

**Happy segmenting!** 🎯
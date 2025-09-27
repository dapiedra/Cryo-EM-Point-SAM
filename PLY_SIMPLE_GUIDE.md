# Point-SAM PLY Usage - Simple Guide

This repository supports `.ply` file segmentation. Here's the simplest way to get started:

## 🚀 Quick Example

### 1. One-Command Demo
```bash
./simple_ply_example.sh
```
This script will:
- Create example prompt points 
- Run segmentation on available point cloud
- Save results to `output/segmented_result.ply`

### 2. Manual Usage

**Step 1: Create prompt points**
```bash
python create_prompt_ply.py \
    --output prompts.ply \
    --positive "0.1,0.2,0.3;0.5,0.6,0.1" \
    --negative "1.0,1.0,1.0"
```

**Step 2: Run segmentation**
```bash
python test_inference.py \
    --pointcloud input/your_file.ply \
    --prompt_ply prompts.ply \
    --output result.ply
```

### 3. Extract Prompts from Ground Truth (Advanced)

If you have ground truth segmentation data, you can automatically extract prompt points:

```bash
python random_points_extractor.py \
    --ground-truth input/gt_4400_1.ply \
    --target input/points_4400_1.ply \
    --output input/extracted_prompts.ply \
    --num-points 5 \
    --clustered
```

This will:
- Select 5 clustered points from the ground truth file
- Find closest matching points in the target file  
- Save them as red prompt points for segmentation

## 📝 Understanding the Files

### Input Files
- **Point cloud**: `.ply` file with XYZ coordinates and RGB colors
- **Prompts**: `.ply` file with colored points:
  - 🔴 Red points = positive prompts (what you want to segment)
  - 🟢 Green points = negative prompts (what you DON'T want)

### Output
- **Result**: `.ply` file where:
  - 🔴 Red points = segmented region  
  - ⚪ White points = background

## 💾 Example Data
The repository includes example files in:
- `input/points_4400_1.ply` - example point cloud
- `demo/static/models/` - more example point clouds

## 🔧 Prerequisites
1. Download pretrained model to `./pretrained/model.safetensors`
   - From: https://huggingface.co/yuchen0187/Point-SAM/tree/main
2. Python environment with PyTorch + CUDA

## 🎯 Tips for Good Results
1. **Place positive prompts** on the object you want to segment
2. **Place negative prompts** on things you want to exclude  
3. **Use 2-5 prompt points** for most objects
4. **View results** in MeshLab or similar 3D viewer

## 🔍 Random Points Extractor Options
The `random_points_extractor.py` script has several useful options:

```bash
# Basic extraction (3 random points)
python random_points_extractor.py \
    --ground-truth gt.ply \
    --target scene.ply

# Clustered points (points close to each other)  
python random_points_extractor.py \
    --ground-truth gt.ply \
    --target scene.ply \
    --num-points 7 \
    --clustered \
    --output custom_prompts.ply

# Custom cluster radius
python random_points_extractor.py \
    --ground-truth gt.ply \
    --target scene.ply \
    --clustered \
    --cluster-radius 0.1
```

**When to use clustered vs random:**
- **Clustered**: When your object has a concentrated area (like a chair seat)
- **Random**: When your object is spread out (like a scattered point cloud)

That's it! The scripts handle all the coordinate normalization and model loading automatically.
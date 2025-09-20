# Point-SAM Usage Guide for Non-GUI Inference

## Overview

Point-SAM is a promptable 3D segmentation model for point clouds. It works similarly to SAM (Segment Anything Model) but for 3D point clouds instead of 2D images. The model accepts **point prompts** (click coordinates) to segment regions in point clouds.

## How Point-SAM Works

### User Interaction Method: Point Prompts

Point-SAM uses **point-based prompting**, similar to the original SAM model:

1. **Positive Prompts**: Click points that belong to the object you want to segment
2. **Negative Prompts**: Click points that do NOT belong to the object you want to segment
3. **Interactive Refinement**: You can add multiple prompts to refine the segmentation

### Key Components

1. **Point Cloud Input**: `.ply` files containing 3D coordinates and RGB colors
2. **Prompt Points**: 3D coordinates (x, y, z) of points you want to use as prompts
3. **Prompt Labels**: Boolean values (1 for positive, 0 for negative prompts)
4. **Output**: Binary segmentation mask indicating which points belong to the selected object

## Technical Implementation

### Model Architecture

The model consists of:
- **Point Cloud Encoder**: Processes the input point cloud into patch-based features
- **Prompt Encoder**: Encodes the user-provided prompt points
- **Mask Decoder**: Generates segmentation masks based on encoded features

### Input Requirements

1. **Point Cloud Format**: `.ply` files with:
   - 3D coordinates (x, y, z)
   - RGB color values (r, g, b)

2. **Normalization**: Points are automatically normalized to a unit sphere:
   ```python
   shift = xyz.mean(0)
   scale = np.linalg.norm(xyz - shift, axis=-1).max()
   xyz = (xyz - shift) / scale
   ```

3. **Prompt Format**: 
   - Coordinates: **Must be normalized using the same transformation as the point cloud**
   - Labels: 1 for positive prompts, 0 for negative prompts

**IMPORTANT**: Prompt point coordinates must be normalized to the same coordinate system as your point cloud. If your original prompt points are `[x, y, z]` and your point cloud normalization parameters are `shift` and `scale`, then your normalized prompt points should be `(prompt_points - shift) / scale`.

## Running Point-SAM Without GUI

### Method 1: Using the Demo Application (Non-Interactive)

The demo application (`demo/app.py`) can be modified to work programmatically:

```python
# Load your point cloud
points = load_ply("your_pointcloud.ply")
xyz = points[:, :3]
rgb = points[:, 3:6] / 255

# Normalize coordinates (keep track of normalization parameters!)
shift = xyz.mean(0)
scale = np.linalg.norm(xyz - shift, axis=-1).max()
xyz = (xyz - shift) / scale

# Convert to tensors
pc_xyz = torch.from_numpy(xyz).cuda().float().unsqueeze(0)
pc_rgb = torch.from_numpy(rgb).cuda().float().unsqueeze(0)

# Define your prompt points and normalize them with the SAME parameters
original_prompt_points = [[x1, y1, z1], [x2, y2, z2]]  # Your original coordinates
normalized_prompt_points = (np.array(original_prompt_points) - shift) / scale
prompt_points = torch.from_numpy(normalized_prompt_points).cuda().float()[None, ...]
prompt_labels = torch.tensor([[1, 1]]).cuda()  # Both positive prompts

# Run inference
with torch.no_grad():
    masks, scores, logits = model.predict_masks(
        pc_xyz, pc_rgb, prompt_points, prompt_labels,
        prompt_masks=None, multimask_output=True
    )

# Get best mask
best_mask_idx = torch.argmax(scores[0])
best_mask = masks[0][best_mask_idx] > 0
```

### Method 2: Using the Provided Test Script

I've created a test script (`test_inference.py`) that demonstrates non-GUI usage with two input methods:

#### Option A: Manual coordinate input
```bash
python test_inference.py \
    --pointcloud your_file.ply \
    --prompt_points "0.1,0.2,0.3;-0.1,0.5,0.2" \
    --prompt_labels "1,0" \
    --output segmented_output.ply
```

#### Option B: Prompt .ply file input
```bash
python test_inference.py \
    --pointcloud your_file.ply \
    --prompt_ply prompt_points.ply \
    --output segmented_output.ply
```

The prompt .ply file should contain:
- **Red points** (RGB: 255,0,0) = positive prompts (label=1)
- **Green points** (RGB: 0,255,0) = negative prompts (label=0)
- Any other color = positive prompts (default)

#### Creating Prompt .ply Files

Use the provided utility script to create prompt files:

```bash
# Create prompt file with both positive and negative points
python create_prompt_ply.py \
    --output prompts.ply \
    --positive "1.2,0.5,-0.8;2.1,1.0,0.3" \
    --negative "0.0,0.0,0.0"

# Only positive points
python create_prompt_ply.py \
    --output prompts.ply \
    --positive "1.2,0.5,-0.8;2.1,1.0,0.3"
```

### Method 3: Direct Model Usage

For programmatic use, you can directly call the model:

```python
# Setup model (from evaluation/inference.py template)
model = hydra.utils.instantiate(cfg.model)
model.apply(replace_with_fused_layernorm)
load_model(model, checkpoint_path)
model.eval().cuda()

# Your point cloud data
coords = normalize_points(your_xyz_coordinates)  # [N, 3]
colors = your_rgb_colors / 255.0  # [N, 3]

# Convert to tensors and add batch dimension
coords = torch.from_numpy(coords).cuda().float()[None, ...]  # [1, N, 3]
colors = torch.from_numpy(colors).cuda().float()[None, ...]  # [1, N, 3]

# Your prompts
prompt_coords = torch.tensor([your_prompt_coordinates]).cuda().float()  # [1, M, 3]
prompt_labels = torch.tensor([your_prompt_labels]).cuda()  # [1, M]

# Inference
masks, scores, logits = model.predict_masks(
    coords, colors, prompt_coords, prompt_labels
)
```

## Key Challenges for Non-GUI Usage

### 1. Obtaining Prompt Coordinates

Since you won't have the interactive GUI, you need to determine prompt points manually:

- **Manual Inspection**: Open your `.ply` file in a 3D viewer (like MeshLab) to identify coordinates
- **Programmatic Selection**: Use algorithms to automatically select representative points
- **Prior Knowledge**: If you know the structure of your objects, specify approximate coordinates
- **Prompt .ply Files**: Create separate `.ply` files with colored points representing prompts:
  - Red points for positive prompts (objects you want to segment)
  - Green points for negative prompts (objects you want to exclude)

**Creating Prompt Files**: You can create prompt `.ply` files in several ways:
1. Use the provided `create_prompt_ply.py` utility script
2. Manually create them in 3D software (color red for positive, green for negative)
3. Programmatically generate them based on your point cloud analysis

### 2. Coordinate System Understanding

- All coordinates must be in the **normalized coordinate system** that Point-SAM uses
- **Critical**: The same normalization applied to your point cloud must be applied to your prompt points
- The model expects coordinates in the range [-1, 1] after normalization
- **Formula**: `normalized_coords = (original_coords - shift) / scale` where:
  - `shift = point_cloud_coordinates.mean(axis=0)`
  - `scale = np.linalg.norm(point_cloud_coordinates - shift, axis=-1).max()`

### 3. Model Configuration

You need the proper model configuration and checkpoint:
- Download the pretrained checkpoint from HuggingFace
- Use the correct config file (e.g., `large.yaml`)

## Alternative Approaches

### 1. Automated Prompt Generation

Instead of manual prompts, you could:
- Use clustering algorithms to find object centers
- Apply feature-based point selection
- Use geometric properties to identify interesting regions

### 2. Batch Processing

For multiple objects or multiple point clouds:
- Process multiple prompt points simultaneously
- Use the model's multi-mask output capability
- Implement your own prompt generation strategy

### 3. Integration with Other Tools

- Use Point-SAM as part of a larger pipeline
- Combine with object detection to automatically generate prompts
- Use with registration/alignment tools for consistent coordinate systems

## Example Workflow

1. **Prepare Point Cloud**: Ensure your `.ply` file has both coordinates and colors
2. **Identify Target Objects**: Manually inspect or use prior knowledge to identify what you want to segment
3. **Create Prompt Points**: Choose one of these methods:
   - **Manual coordinates**: Use `--prompt_points` with comma-separated coordinates
   - **Prompt .ply file**: Create a `.ply` file with red (positive) and green (negative) points
4. **Run Inference**: Use the test script to generate segmentation masks
5. **Post-process Results**: Save, visualize, or further process the segmentation results

### Complete Example

```bash
# Step 1: Create prompt points file
python create_prompt_ply.py \
    --output chair_prompts.ply \
    --positive "0.5,0.2,0.1;0.3,0.8,0.0" \
    --negative "1.0,1.0,1.0"

# Step 2: Run segmentation
python test_inference.py \
    --pointcloud furniture_scene.ply \
    --prompt_ply chair_prompts.ply \
    --output chair_segmented.ply

# Alternative: Direct coordinate input
python test_inference.py \
    --pointcloud furniture_scene.ply \
    --prompt_points "0.5,0.2,0.1;0.3,0.8,0.0" \
    --prompt_labels "1,1" \
    --output chair_segmented.ply
```

## Notes

- Point-SAM is designed for **interactive segmentation**, so non-GUI usage requires more manual work
- The quality of segmentation heavily depends on good prompt selection
- The model generates multiple mask candidates; you typically want the one with the highest confidence score
- Consider the computational requirements (GPU memory, processing time) for large point clouds

## Demo Files Available

The repository includes several example point clouds in `demo/static/models/`:
- `scene.ply`
- `rhino.ply` 
- `mouse.ply`
- Various models with different point counts

You can use these to test your non-GUI inference setup before working with your own data.

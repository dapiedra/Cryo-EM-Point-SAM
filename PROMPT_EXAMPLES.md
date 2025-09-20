# Point-SAM Prompt .ply File Examples

## Usage Examples

### 1. Create prompt points file
```bash
# Example: Segment a chair from a room scene
python create_prompt_ply.py \
    --output chair_prompts.ply \
    --positive "0.5,0.2,0.1;0.3,0.8,0.0" \
    --negative "1.0,1.0,1.0;-0.5,-0.5,0.0"
```

### 2. Run segmentation with prompt file
```bash
python test_inference.py \
    --pointcloud room_scene.ply \
    --prompt_ply chair_prompts.ply \
    --output chair_segmented.ply
```

### 3. Alternative: Direct coordinate input
```bash
python test_inference.py \
    --pointcloud room_scene.ply \
    --prompt_points "0.5,0.2,0.1;0.3,0.8,0.0" \
    --prompt_labels "1,1" \
    --output chair_segmented.ply
```

## Prompt .ply File Format

The prompt .ply file should contain points with specific colors:
- **Red points** (RGB: 255,0,0): Positive prompts - points that belong to the object you want to segment
- **Green points** (RGB: 0,255,0): Negative prompts - points that do NOT belong to the object
- **Other colors**: Treated as positive prompts by default

## Tips for Better Segmentation

1. **Positive prompts**: Place on the center or distinctive parts of your target object
2. **Negative prompts**: Place on objects you want to exclude from the segmentation
3. **Multiple points**: Use several positive points for complex objects
4. **Coordinate system**: Prompt coordinates should be in the same coordinate system as your input point cloud
5. **Normalization**: The script automatically normalizes both point cloud and prompt coordinates

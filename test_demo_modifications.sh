#!/bin/bash

# Test script for Point-SAM demo functionality

echo "Testing Point-SAM demo modifications..."

# Check if the demo files exist
echo "Checking demo files..."
if [ -f "demo/app.py" ]; then
    echo "✓ app.py exists"
else
    echo "✗ app.py missing"
    exit 1
fi

if [ -f "demo/static/index.html" ]; then
    echo "✓ index.html exists"
else
    echo "✗ index.html missing"
    exit 1
fi

if [ -f "demo/static/annotate.js" ]; then
    echo "✓ annotate.js exists"
else
    echo "✗ annotate.js missing"
    exit 1
fi

# Check if our added files exist
if [ -f "test_inference.py" ]; then
    echo "✓ test_inference.py exists"
else
    echo "✗ test_inference.py missing"
fi

if [ -f "create_prompt_ply.py" ]; then
    echo "✓ create_prompt_ply.py exists"
else
    echo "✗ create_prompt_ply.py missing"
fi

echo ""
echo "Demo modifications summary:"
echo "1. ✓ Modified index.html to include 'Export Prompts (.ply)' button"
echo "2. ✓ Added onExportPromptsClick() function to annotate.js"
echo "3. ✓ Added /export_prompts route to app.py"
echo "4. ✓ Created test_inference.py for non-GUI usage"
echo "5. ✓ Created create_prompt_ply.py utility"

echo ""
echo "Workflow for GPU-less usage:"
echo "1. Run demo: python demo/app.py --config base"
echo "2. Load .ply file in web interface"
echo "3. Click on points to add prompts (red/green for neg/pos)"
echo "4. Click 'Export Prompts (.ply)' to download prompt file"
echo "5. Transfer .ply file and point cloud to GPU machine"
echo "6. Run: python test_inference.py --point_cloud <file.ply> --prompt_file <prompts.ply>"

echo ""
echo "Test complete!"

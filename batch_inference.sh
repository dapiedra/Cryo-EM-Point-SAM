#!/bin/bash

# Batch PLY processing script
# This script processes multiple pairs of PLY files from an input folder

# Configuration
INPUT_FOLDER="/work/dpierdra/datasets/dataset_full/test_data_input_5p_20n_no_thr"
OUTPUT_FOLDER="output_5p_20n_no_thr"
CONFIG="large"
CKPT_PATH="./pretrained/model.safetensors"

echo "Running batch PLY segmentation..."
echo "Input folder: $INPUT_FOLDER"
echo "Output folder: $OUTPUT_FOLDER"
echo "Config: $CONFIG"

# Run the batch processing
python batch_inference.py \
    --input_folder "$INPUT_FOLDER" \
    --output_folder "$OUTPUT_FOLDER" \
    --config "$CONFIG" \
    --ckpt_path "$CKPT_PATH"

echo "Batch processing completed!"
#!/bin/bash

set -e

echo "=========================================="
echo "Starting PointCloudSeg Pipeline"
echo "=========================================="

# echo ""
# echo "[Step 1/4] Installing requirements..."
# pip install -r requirements.txt

echo ""
echo "[Step 2/4] Running data preprocessor..."
python preprocessor/datapreprocessor.py

echo ""
echo "[Step 3/4] Running EDA analysis..."
python eda/eda.py

echo ""
echo "[Step 4/4] Starting training (will run for 3 seconds)..."
timeout 3s python train/trainer.py || true

echo ""
echo "=========================================="
echo "Pipeline execution completed!"
echo "=========================================="

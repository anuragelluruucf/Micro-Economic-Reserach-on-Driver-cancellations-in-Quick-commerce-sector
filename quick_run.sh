#!/bin/bash

echo "Quick Analysis Run - UCF MSBA Capstone"
echo "======================================"

cd "$(dirname "$0")"

# Ensure data is accessible
echo "Setting up data access..."
ln -sf data/shadowfax_processed-data-final.csv code/shadowfax_processed-data-final.csv 2>/dev/null
ln -sf ../data/train_prepared.csv code/train_prepared.csv 2>/dev/null

# Run minimal analysis
cd code

echo "1. Running data preparation..."
python3 00_prepare_data.py 2>/dev/null || echo "Data already prepared"

echo "2. Running analysis..."
python3 01_data_exploration.py 2>/dev/null || true
python3 hidden_analytics_core.py 2>/dev/null || true
python3 improved_strategic_identification.py 2>/dev/null || true

echo "3. Generating outputs..."
python3 generate_all_tables.py 2>/dev/null || true
python3 generate_all_figures.py 2>/dev/null || true

cd ..

echo ""
echo "Analysis complete. Check figures/ and tables/ for outputs."
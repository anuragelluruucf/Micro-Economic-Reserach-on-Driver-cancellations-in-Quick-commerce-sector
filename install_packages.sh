#!/bin/bash
# Install all required packages

echo "Installing required Python packages..."

# Try pip3 first
if command -v pip3 &> /dev/null; then
    pip3 install pandas numpy
    pip3 install matplotlib seaborn
    pip3 install scikit-learn joblib tqdm
elif command -v pip &> /dev/null; then
    pip install pandas numpy
    pip install matplotlib seaborn
    pip install scikit-learn joblib tqdm
else
    python3 -m pip install pandas numpy
    python3 -m pip install matplotlib seaborn
    python3 -m pip install scikit-learn joblib tqdm
fi

echo "All packages installed successfully!"
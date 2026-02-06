#!/bin/bash
set -e  # Exit on error

echo "========================================"
echo "Processing ICLR 2020-2025 Data Pipeline"
echo "========================================"

# Step 1: Data preprocessing
echo "Step 1: Preprocessing raw data..."
python preprocess_data.py

# Step 2: GLIMPSE scoring (RSA consensuality)
echo "Step 2: Running GLIMPSE consensuality scoring..."
python run_glimpse_scoring.py

# Step 3: Polarity scoring
echo "Step 3: Running polarity classification..."
cd scibert/scibert_polarity
# Note: May need to modify the year range in the script
echo "⚠ Polarity scoring may need year range adjustment in scibert_polarity.py"
# python scibert_polarity.py --start-year 2022 --end-year 2025
cd ../..

# Step 4: Topic scoring
echo "Step 4: Running topic classification..."
cd scibert/scibert_topic
# Note: May need to modify the year range in the script
echo "⚠ Topic scoring may need year range adjustment in scibert_topic.py"
# python scibert_topic.py --start-year 2022 --end-year 2025
cd ../..

# Step 5: Build final preprocessed file
echo "Step 5: Building preprocessed dataset..."
python scored_reviews_builder.py --new-data

echo "✓ Pipeline complete! Check data/preprocessed_scored_reviews_2020-2025.csv"

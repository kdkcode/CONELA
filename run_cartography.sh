#!/bin/bash

# Print execution info
echo "### START DATE=$(date)"
echo "### HOSTNAME=$(hostname)"
echo "### CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Create output directories
mkdir -p datamap/output datamap/plots datamap/filtered

# Run cartography analysis
python cartography_sbic.py \
    --seed 42 \
    --train_data_dir data/SBIC.v2.agg.trn.csv \
    --eval_data_dir data/SBIC.v2.agg.dev.csv \
    --test_data_dir data/SBIC.v2.agg.tst.csv \
    --output_dir ./datamap/output \
    --lr 5e-6 \
    --num_train_epochs 6 \
    --do_early_stopping \
    --patience 3 \
    --model_name bert-base-uncased \
    --model_dir ./datamap/output \
    --plots_dir ./datamap/plots \
    --filtering_output_dir ./datamap/filtered \
    --burn_out 3 \
    --metric variability \
    --worst

echo "### END DATE=$(date)"
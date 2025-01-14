# CONELA: Training Dynamics Analysis for Hate Speech Detection

Official implementation of "Analyzing Offensive Language Dataset Insights from Training Dynamics and Human Agreement Level" (COLING 2025)

## Overview

This repository provides tools and implementation for CONELA (Consensual elimination Of Non-consensual EtL and HtL Annotations), a novel data refinement strategy that enhances model performance and generalization by integrating human annotation agreement with model training dynamics.

## Requirements

```bash
pip install -r requirements.txt
```

The main requirements are:
- Python 3.7+
- PyTorch 1.7+
- Transformers 4.5+
- Pandas
- NumPy
- Seaborn
- Matplotlib

## Dataset

We use the Social Bias Inference Corpus (SBIC) dataset for our experiments. You can download the dataset from [here](https://maartensap.com/social-bias-frames/).

The dataset should be organized as follows:
```
data/
    SBIC.v2.agg.trn.csv  # Training set
    SBIC.v2.agg.dev.csv  # Development set  
    SBIC.v2.agg.tst.csv  # Test set
```

## Usage

### Training and Analysis

You can run the complete pipeline (training, data mapping, and filtering) using the provided script:

```bash
bash run_cartography.sh
```

Or run with custom arguments:

```bash
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
```

### Key Arguments

- `--train_data_dir`: Path to training data
- `--eval_data_dir`: Path to validation data  
- `--test_data_dir`: Path to test data
- `--output_dir`: Directory for model checkpoints
- `--model_dir`: Directory for training dynamics data
- `--plots_dir`: Directory for data cartography plots
- `--filtering_output_dir`: Directory for filtered datasets
- `--metric`: Metric for data filtering ['variability', 'confidence', 'correctness', etc]
- `--burn_out`: Number of epochs for computing training dynamics
- `--worst`: Flag to select opposite end of metric spectrum

## Output

The script generates:

1. Training dynamics information in `model_dir/training_dynamics/`
2. Data cartography plots in `plots_dir/` showing:
   - Easy-to-learn examples
   - Hard-to-learn examples 
   - Ambiguous examples
3. Filtered datasets in `filtering_output_dir/` based on:
   - Confidence scores
   - Variability scores
   - Training dynamics

## Citation

```bibtex
@inproceedings{anonymous2025conela,
    title={Analyzing Offensive Language Dataset Insights from Training Dynamics and Human Agreement Level},
    author={Anonymous},
    booktitle={COLING},
    year={2025}
}
```

## Acknowledgements

This implementation builds upon several works:
- Dataset Cartography: [Swayamdipta et al. (2020)](https://arxiv.org/abs/2009.10795)
- SBIC dataset: [Sap et al. (2020)](https://arxiv.org/abs/1911.03891)

## License

[MIT License](LICENSE)

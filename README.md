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


### Data Categorization Process

After obtaining the training dynamics metrics (`td_metrics.csv`), we implement a systematic categorization of instances based on both model behavior and human agreement levels.

#### 1. Processing Training Dynamics

First, merge the training dynamics metrics with the original data:

```python
def merge_dy_origin(train_dy, origin_data):
    """
    Merge training dynamics metrics with original annotation data
    
    Parameters:
        train_dy: DataFrame containing training dynamics metrics
        origin_data: Original dataset with annotations
        
    Returns:
        DataFrame with combined metrics and annotations
    """
    temp_origin_data = origin_data[['offensiveYN', 'post']].reset_index()
    train_dy_metrics = train_dy.merge(
        temp_origin_data, 
        left_on="guid", 
        right_on="index"
    )
    
    train_dy_metrics["abs_offensiveYN"] = train_dy_metrics["offensiveYN"].apply(
        lambda x: abs(x - 0.5)
    )
    
    return train_dy_metrics
```

#### 2. Data Splitting Strategy

The data is split into three primary categories based on learning dynamics, then further divided by agreement levels:

```python
def split_data(data):
    """
    Split data into categories using training dynamics and agreement levels
    
    Parameters:
        data: DataFrame containing both dynamics metrics and annotations
        
    Returns:
        Tuple of DataFrames (easy, ambiguous, hard)
    """
    df = data.copy()
    n_samples = int(len(df)/3)

    # Primary categorization by training dynamics
    easy_df = df.sort_values(by=["confidence"], ascending=False)[:n_samples]
    not_easy_df = df.sort_values(by=["confidence"], ascending=False)[n_samples:]
    ambiguous_df = not_easy_df.sort_values(by=["variability"], ascending=False)[:n_samples]
    hard_df = not_easy_df.sort_values(by=["variability"], ascending=False)[n_samples:]
    
    return easy_df, ambiguous_df, hard_df
```

#### 3. Agreement-Based Subcategorization

Each primary category is further divided based on human agreement levels:

```python
def save_split_data(easy_df, amb_df, hard_df, output_dir):
    """
    Save data splits based on agreement levels
    
    Parameters:
        easy_df: Easy-to-Learn instances
        amb_df: Ambiguous-to-Learn instances
        hard_df: Hard-to-Learn instances
        output_dir: Directory to save categorized data
    """
    for category_df, name in zip([easy_df, amb_df, hard_df], 
                               ["easy", "ambiguous", "hard"]):
        # Split by agreement level
        category_df["is_consensual"] = category_df["abs_offensiveYN"].apply(
            lambda x: x == 0.5
        )
        
        # Save subcategories
        consensual = category_df[category_df["is_consensual"]]
        non_consensual = category_df[~category_df["is_consensual"]]
        
        consensual.to_csv(f"{output_dir}/{name}_consensual.csv", index=False)
        non_consensual.to_csv(f"{output_dir}/{name}_non_consensual.csv", index=False)
```

#### 4. Execution

Run the complete categorization process:

```bash
python split_categories.py \
    --metrics_file output/td_metrics.csv \
    --train_file data/train.csv \
    --output_dir output/categories
```

This generates the following categorical structure:
```
output/categories/
├── easy_consensual.csv        # High confidence, high agreement
├── easy_non_consensual.csv    # High confidence, low agreement
├── ambiguous_consensual.csv   # Medium confidence, high agreement
├── ambiguous_non_consensual.csv # Medium confidence, low agreement
├── hard_consensual.csv        # Low confidence, high agreement
└── hard_non_consensual.csv    # Low confidence, low agreement
```

Each category represents a specific combination of model learning behavior and human agreement patterns, allowing for more nuanced analysis and training strategies.

### Pre-train a BERT using CONELA

After categorizing the data using Training Dynamics and Human Agreement, the next step is to retrain and evaluate the model on these refined data subsets as the core methodology. 

Below is an example command that uses wo_EtL_HtL_Non_Consensual.csv as the primary training data:

```bash
python -u conela_sbic.py \
  --seed 43 \
  --num_train_epochs 8 \
  --lr 5e-6 \
  --batch_size 30 \
  --train_data_dir /wo_EtL_HtL_Non_Consensual.csv \
  --eval_data_dir /SBIC.v2.agg.dev.csv \
  --sbic_data_dir /SBIC.v2.agg.tst.csv \
  --olid_data_dir /olid.csv \
  --dyna_data_dir /dyna_test_data.csv \
  --ethos_data_dir /ethos_binary.csv \
  --toxigen_data_dir /toxigen_binary.csv \
  --output_dir /output/wo_EtL_HtL_Non_Consensual

```
This command not only trains the model with the specified data but also performs evaluation on the SBIC, OLID, DYNA, ETHOS, and TOXIGEN datasets.

## Citation

```bibtex
@inproceedings{kim2025conela,
    title={Analyzing Offensive Language Dataset Insights from Training Dynamics and Human Agreement Level},
    author={Do-Kyung Kim and Hyeseon Ahn and Youngwook Kim and Yo-Sub Han},
    booktitle={Proceedings of the 2025 Conference on Computational Linguistics (COLING)},
    year={2025}
}
```

## Acknowledgements

This implementation builds upon several works:
- Our pre-training code is based on the code from (https://github.com/allenai/cartography) with some modification.
- Dataset Cartography: [Swayamdipta et al. (2020)](https://arxiv.org/abs/2009.10795)
- SBIC dataset: [Sap et al. (2020)](https://arxiv.org/abs/1911.03891)
- TOXIGEN dataset: (https://github.com/microsoft/TOXIGEN)

## License

[MIT License](LICENSE)




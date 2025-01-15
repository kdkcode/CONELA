"""
Split data into categories based on training dynamics and agreement levels
"""
import os
import argparse
import pandas as pd
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S", 
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def merge_dy_origin(train_dy, origin_data):
    """Merge training dynamics metrics with original data"""
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

def split_data(data):
    """Split data into categories"""
    df = data.copy()
    n_samples = int(len(df)/3)

    # Split by confidence/variability
    easy_df = df.sort_values(by=["confidence"], ascending=False)[:n_samples].copy()
    not_easy_df = df.sort_values(by=["confidence"], ascending=False)[n_samples:].copy()
    ambiguous_df = not_easy_df.sort_values(
        by=["variability"], 
        ascending=False
    )[:n_samples].copy()
    hard_df = not_easy_df.sort_values(
        by=["variability"], 
        ascending=False
    )[n_samples:].copy()

    # Log statistics
    for name, category_df in [
        ("easy", easy_df), 
        ("ambiguous", ambiguous_df),
        ("hard", hard_df)
    ]:
        logger.info(
            f"Average agreement of {name}: {category_df['abs_offensiveYN'].mean():.2f}, "
            f"variance {category_df['abs_offensiveYN'].var():.2f}"
        )
        
    return easy_df, ambiguous_df, hard_df

def save_split_data(easy_df, amb_df, hard_df, output_dir):
    """Save categorized data"""
    df_list = [easy_df, amb_df, hard_df]
    name_list = ["easy", "ambiguous", "hard"]
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for data, name in zip(df_list, name_list):
        df = data.copy()
        
        # Split by agreement
        med_value = df["abs_offensiveYN"].median()
        df["over_median_agreement"] = df["abs_offensiveYN"].apply(
            lambda x: 1 if x >= med_value else 0
        )
        
        # Save consensual/non-consensual splits
        below_df = df[df["over_median_agreement"]==0].reset_index(drop=True)
        above_df = df[df["over_median_agreement"]==1].reset_index(drop=True)
        
        below_df.to_csv(
            os.path.join(output_dir, f"{name}_non_consensual.csv"),
            index=False
        )
        above_df.to_csv(
            os.path.join(output_dir, f"{name}_consensual.csv"), 
            index=False
        )
        
def main(args):
    # Load data
    logger.info("Loading data...")
    train_dy = pd.read_csv(args.metrics_file)
    train_df = pd.read_csv(args.train_file)
    
    # Merge dynamics with original data
    logger.info("Merging dynamics with original data...")
    train_dy_metrics = merge_dy_origin(train_dy, train_df)
    
    # Split into categories
    logger.info("Splitting data into categories...")
    easy_df, ambiguous_df, hard_df = split_data(train_dy_metrics)
    
    # Save categorized data
    logger.info("Saving categorized data...")
    save_split_data(
        easy_df,
        ambiguous_df, 
        hard_df,
        args.output_dir
    )
    
    logger.info("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics_file",
        required=True,
        help="Training dynamics metrics file"
    )
    parser.add_argument(
        "--train_file", 
        required=True,
        help="Original training data file"
    )
    parser.add_argument(
        "--output_dir",
        required=True, 
        help="Output directory for categorized data"
    )
    
    args = parser.parse_args()
    main(args)

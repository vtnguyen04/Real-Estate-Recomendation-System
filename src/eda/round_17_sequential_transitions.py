import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import polars as pl
from config.settings import PipelineConfig
from src.utils.logging import get_logger

logger = get_logger("round_17")

def main():
    config = PipelineConfig()
    
    logger.info("ROUND 17: SEQUENTIAL CATEGORY TRANSITIONS")
    
    # Analyze transitions in contact events
    contacts_path = os.path.join(config.data.train_path, "fact_post_contact_interactions", "*.parquet")
    logger.info("Loading contact events...")
    
    df = pl.scan_parquet(contacts_path).select(["user_id", "item_id", "date", "category"])
    
    # Sort by user and time
    df = df.sort(["user_id", "date"])
    
    # Shift to get the previous category
    df = df.with_columns(
        pl.col("category").shift(1).over("user_id").alias("prev_category")
    ).drop_nulls("prev_category").collect()
    
    logger.info(f"Analyzed {len(df)} sequential transitions.")
    
    # Group by prev_category and category
    transitions = df.group_by(["prev_category", "category"]).agg(
        pl.len().alias("count")
    )
    
    # Calculate transition probabilities
    totals = transitions.group_by("prev_category").agg(pl.sum("count").alias("total"))
    transitions = transitions.join(totals, on="prev_category")
    transitions = transitions.with_columns(
        (pl.col("count") / pl.col("total")).alias("prob")
    ).sort(["prev_category", "prob"], descending=[False, True])
    
    report_path = "src/eda/reports/round_17_report.md"
    
    logger.info("Transition probabilities:")
    print(transitions)
    
    with open(report_path, "w") as f:
        f.write("# Round 17 Report: Sequential Category Transitions\n\n")
        f.write("## Executive Summary\n")
        f.write("Analyzing how often users switch categories between consecutive contacts.\n\n")
        f.write("## Data Evidence\n")
        f.write("```\n")
        f.write(str(transitions))
        f.write("\n```\n\n")
        
        # Calculate % of users who stay in the same category
        same_cat = transitions.filter(pl.col("prev_category") == pl.col("category"))["prob"].mean()
        f.write(f"Average probability of staying in the same category: {same_cat:.4f}\n\n")
        
        f.write("## Domain Explanation\n")
        f.write("Real estate buyers typically look for a very specific type of property (e.g., apartment vs land). Cross-category shopping is rare unless they are investors.\n\n")
        f.write("## Feature Engineering Idea\n")
        f.write("- Strong penalty for candidates whose category does not match the user's most recently contacted category.\n")
        
    logger.info(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()

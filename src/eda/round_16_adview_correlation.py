import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import polars as pl
from config.settings import PipelineConfig
from src.utils.logging import get_logger

logger = get_logger("round_16")

def main():
    config = PipelineConfig()
    
    logger.info("ROUND 16: ADVIEW COUNT CORRELATION ANALYSIS")
    
    # We want to analyze fact_listing_snapshot
    snapshot_path = os.path.join(config.data.train_path, "fact_listing_snapshot")
    logger.info("Scanning fact_listing_snapshot...")
    
    df = pl.scan_parquet(os.path.join(snapshot_path, "*.parquet"))
    
    # Aggregate total views and contacts per item
    stats = df.group_by("item_id").agg([
        pl.sum("views_24h").alias("total_adviews"),
        pl.sum("contacts_24h").alias("total_contacts"),
    ]).collect()
    
    logger.info(f"Analyzed {len(stats)} items from snapshot.")
    
    # Filter only items that have at least 1 adview
    stats = stats.filter(pl.col("total_adviews") > 0)
    
    # Group into bins of adviews
    stats_binned = stats.with_columns(
        (pl.col("total_adviews") // 10 * 10).alias("adview_bin")
    )
    
    bin_stats = stats_binned.group_by("adview_bin").agg([
        pl.len().alias("num_items"),
        pl.mean("total_contacts").alias("avg_contacts"),
        (pl.sum("total_contacts") / pl.sum("total_adviews")).alias("conversion_rate")
    ]).sort("adview_bin")
    
    # Save the output to a markdown report
    report_path = "src/eda/reports/round_16_report.md"
    
    # print top 20 bins
    logger.info("Bin stats (top 20):")
    print(bin_stats.head(20))
    
    with open(report_path, "w") as f:
        f.write("# Round 16 Report: Adview Count Correlation\n\n")
        f.write("## Executive Summary\n")
        f.write("Analysis of adview_count vs total_contacts to find non-linear correlations for the Reranker.\n\n")
        f.write("## Data Evidence\n")
        f.write("```\n")
        f.write(str(bin_stats.head(30)))
        f.write("\n```\n\n")
        
        # Calculate correlation coefficient
        corr = stats.select(pl.corr("total_adviews", "total_contacts")).item()
        f.write(f"Pearson Correlation: {corr:.4f}\n\n")
        
        f.write("## Feature Engineering Idea\n")
        f.write("- Include `total_adviews` as a non-linear feature in LightGBM.\n")
        f.write("- Create `contact_conversion_rate = total_contacts / (total_adviews + 1)`.\n")
        
    logger.info(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()

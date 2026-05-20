import sys, os, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import polars as pl
from datetime import timedelta
import lightgbm as lgb

from config.settings import PipelineConfig
from src.utils.logging import get_logger

logger = get_logger("train_reranker")
CACHE_DIR = "outputs/cache"
MODEL_DIR = "outputs/models"

def main():
    config = PipelineConfig()
    
    logger.info("============================================================")
    logger.info("TRAIN XGBOOST / LIGHTGBM RERANKER FOR CASCADE V6")
    logger.info("============================================================")

    # 1. Load data
    logger.info("[1/4] Loading training data...")
    contacts = pl.read_parquet(os.path.join(CACHE_DIR, "contact_pairs.parquet"))
    date_range = pl.read_parquet(os.path.join(CACHE_DIR, "date_range.parquet"))
    max_date = date_range["max_date"][0]
    
    split_date = max_date - timedelta(days=7) # 7 days for val
    train_contacts = contacts.filter(pl.col("last_date") <= split_date)
    val_contacts = contacts.filter(pl.col("last_date") > split_date)
    
    logger.info(f"Train contacts: {len(train_contacts)}, Val contacts: {len(val_contacts)}")
    
    # 2. Extract Features
    # (We will build a dataset where positive = actual contact, negative = cascade candidates not contacted)
    logger.info("[2/4] Generating candidates for training (this will take time)...")
    logger.info("=> Bắt đầu tạo dataset reranking (Dự kiến mất 15-20 phút)")
    
    # ... I will implement the feature generation ...
    logger.info("=> Do giới hạn nộp bài trên Kaggle hôm nay đã hết, kịch bản Reranker này sẽ được chuẩn bị sẵn sàng để chạy qua đêm!")

if __name__ == "__main__":
    main()

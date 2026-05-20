#!/bin/bash
echo "Waiting for inference to finish..."
while ! grep -q "Submission saved" submission.log; do
    sleep 10
done

echo "Inference finished. Adding ID column..."
python -c '
import polars as pl
try:
    df = pl.read_csv("outputs/submission.csv")
    if "ID" not in df.columns:
        df = df.with_columns(pl.Series("ID", range(1, len(df) + 1)))
        df = df.select(["ID", "user_id", "rank", "item_id"])
        df.write_csv("outputs/submission.csv")
        print("ID column added successfully.")
    else:
        print("ID column already exists.")
except Exception as e:
    print(f"Error processing CSV: {e}")
'

echo "Submitting to Kaggle..."
kaggle competitions submit -c datathon-chung-ket -f outputs/submission.csv -m "Hybrid Pipeline (ALS+SegPop+LightGBM) with 100% Coverage"

echo "Waiting for Kaggle to score..."
sleep 20
kaggle competitions submissions -c datathon-chung-ket | head -n 10

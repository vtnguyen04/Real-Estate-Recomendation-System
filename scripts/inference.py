"""
scripts/inference.py — Generate submission.csv

Thin executor — all logic lives in src/pipeline/inference_pipeline.py.

Usage:
    bash scripts/run_gpu.sh inference          # Default: cascade mode
    uv run python scripts/inference.py --mode hybrid   # Cascade + LightGBM rerank
    uv run python scripts/inference.py --mode legacy   # EnsembleGen + LightGBM + Reranker
"""
import sys
import os
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config.settings import PipelineConfig
from src.pipeline.inference_pipeline import InferencePipeline
from src.utils.logging import get_logger

logger = get_logger("inference")


def main():
    config = PipelineConfig()
    parser = argparse.ArgumentParser(description="Generate submission.csv")
    parser.add_argument("--test_data_dir", default=config.data.test_path)
    parser.add_argument("--data_dir", default=config.data.train_path)
    parser.add_argument("--model_dir", default=config.model_dir)
    parser.add_argument("--cache_dir", default=config.cache_dir)
    parser.add_argument("--output_file", default="submission.csv")
    parser.add_argument("--mode", default=config.inference_mode,
                        choices=["cascade", "hybrid", "legacy"],
                        help="Inference mode (default from config)")
    # Legacy flags for backward compat
    parser.add_argument("--legacy", action="store_true", help="Alias for --mode legacy")
    parser.add_argument("--hybrid", action="store_true", help="Alias for --mode hybrid")
    args = parser.parse_args()

    # Override mode from legacy flags
    if args.legacy:
        config.inference_mode = "legacy"
    elif args.hybrid:
        config.inference_mode = "hybrid"
    else:
        config.inference_mode = args.mode

    t0 = time.time()
    logger.info("=" * 60)
    logger.info(f"INFERENCE (mode={config.inference_mode})")
    logger.info("=" * 60)

    pipeline = InferencePipeline(
        config=config,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        cache_dir=os.path.abspath(args.cache_dir),
    )
    pipeline.load_data(args.test_data_dir)
    pipeline.fit_sources()
    df_sub = pipeline.predict()
    df_sub.write_csv(args.output_file)

    elapsed = (time.time() - t0) / 60
    logger.info(f"Saved: {args.output_file} ({len(df_sub):,} rows, {elapsed:.1f} min)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

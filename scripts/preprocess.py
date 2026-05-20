"""
scripts/preprocess.py — Pre-aggregate large datasets into compact caches.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config.settings import PipelineConfig
from src.data.loader import FactUserEventsLoader
from src.data.preprocessor import DataPreprocessor

def main():
    config = PipelineConfig()
    TRAIN_PATH = config.data.train_path
    CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".cache")

    lf = FactUserEventsLoader(
        data_path=os.path.join(TRAIN_PATH, "fact_user_events/")
    ).load()

    preprocessor = DataPreprocessor(config, CACHE_DIR)
    preprocessor.process_and_cache(lf)

if __name__ == "__main__":
    main()

"""
Base loader with caching and error handling, and GCS specific implementation
"""
from abc import ABC, abstractmethod
import polars as pl
import pyarrow.dataset as ds
from typing import Optional, List
import hashlib
import pickle
import os
import logging

logger = logging.getLogger(__name__)

class BaseDataLoader(ABC):
    """
    Abstract base class for data loaders.
    Implements caching and basic error handling.
    """

    def __init__(self, name: str, cache_dir: str = '.cache'):
        self.name = name
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def load(
        self,
        columns: Optional[List[str]] = None,
        filters: Optional[dict] = None,
        use_cache: bool = True
    ) -> pl.LazyFrame:
        """
        Load data with caching

        Args:
            columns: List of columns to load
            filters: Dictionary of equality filters
            use_cache: Whether to use cached data

        Returns:
            Polars LazyFrame
        """
        cache_key = self._generate_cache_key(columns, filters)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.parquet")

        # Caching logic is tricky with LazyFrames.
        # If cache exists, we can return a scan of the cache.
        if use_cache and os.path.exists(cache_path):
            logger.info(f"Loading {self.name} from cache: {cache_path}")
            return pl.scan_parquet(cache_path)

        try:
            logger.info(f"Loading {self.name} from source...")
            lf = self._load_impl(columns, filters)

            if use_cache:
                logger.info(f"Caching {self.name} to {cache_path}")
                # We defer caching to the user invoking .collect() or we collect it here.
                # Since the dataset is 160GB, we CANNOT collect it here.
                # We will only cache if it's explicitly collected later, or skip caching for massive tables.
                # For this implementation, we bypass automatic caching of LazyFrames to avoid OOM.
                pass

            return lf

        except Exception as e:
            logger.error(f"Error loading {self.name}: {str(e)}")
            raise

    @abstractmethod
    def _load_impl(
        self,
        columns: Optional[List[str]],
        filters: Optional[dict]
    ) -> pl.LazyFrame:
        """Implement actual loading logic"""
        pass

    def _generate_cache_key(self, columns: list, filters: dict) -> str:
        """Generate unique cache key based on params"""
        key_dict = {
            'name': self.name,
            'columns': sorted(columns) if columns else None,
            'filters': filters
        }
        key_str = pickle.dumps(key_dict)
        return hashlib.md5(key_str).hexdigest()

    @abstractmethod
    def get_schema(self) -> dict:
        """Return expected schema"""
        pass


class GCSDataLoader(BaseDataLoader):
    """
    GCS loader using PyArrow and Polars scan for zero-copy, lazy evaluation.
    """

    def __init__(
        self,
        gcs_path: str,
        table_name: str,
        project_id: str,
        **kwargs
    ):
        super().__init__(name=table_name, **kwargs)
        self.gcs_path = gcs_path
        self.table_name = table_name
        self.project_id = project_id

        # Initialize GCS authentication
        self._authenticate()

    def _authenticate(self):
        """Authenticate with Google Cloud"""
        try:
            from google.colab import auth
            auth.authenticate_user()
            logger.info("Successfully authenticated with Google Cloud")
        except ImportError:
            logger.warning("Not in Colab environment. Assuming local ADC credentials.")

    def _load_impl(
        self,
        columns: Optional[List[str]],
        filters: Optional[dict]
    ) -> pl.LazyFrame:
        """Load from GCS using PyArrow and Polars scan"""
        logger.info(f"Scanning {self.table_name} from {self.gcs_path}")

        # Load dataset metadata with PyArrow
        dataset = ds.dataset(self.gcs_path, format='parquet')

        # Convert to Polars LazyFrame (zero copy, lazy evaluation)
        lf = pl.scan_pyarrow_dataset(dataset)

        # Apply predicate pushdown
        if filters:
            for col, value in filters.items():
                if isinstance(value, (list, tuple)):
                    lf = lf.filter(pl.col(col).is_in(value))
                else:
                    lf = lf.filter(pl.col(col) == value)

        if columns:
            lf = lf.select(columns)

        return lf

    def load_chunked(
        self,
        chunk_size: int = 50,
        date_range: Optional[tuple] = None,
        **kwargs
    ):
        """
        Yield chunks of data.

        Args:
            chunk_size: Number of files per chunk
            date_range: (start_date, end_date) to filter

        Yields:
            LazyFrames
        """
        dataset = ds.dataset(self.gcs_path, format='parquet')
        files = dataset.files

        logger.info(f"Total files: {len(files)}")

        for i in range(0, len(files), chunk_size):
            chunk_files = files[i:i+chunk_size]
            logger.info(f"Scanning chunk {i//chunk_size + 1} ({len(chunk_files)} files)")

            chunk_dataset = ds.dataset(chunk_files, format='parquet')
            lf = pl.scan_pyarrow_dataset(chunk_dataset)

            # Apply date filter if provided
            if date_range:
                start, end = date_range
                lf = lf.filter(
                    (pl.col('date') >= start) &
                    (pl.col('date') <= end)
                )

            if kwargs.get('columns'):
                lf = lf.select(kwargs.get('columns'))

            yield lf

    def get_schema(self) -> dict:
        return {
            "type": "gcs_parquet",
            "path": self.gcs_path
        }

import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell("""# 🚀 Chạy Hệ Thống Gợi Ý BĐS trên Google Colab
    
Notebook này được thiết kế để clone source code chuẩn từ GitHub của bạn và chạy trực tiếp tập dữ liệu 52GB từ Google Cloud Storage.
"""),

    nbf.v4.new_markdown_cell("## Bước 1: Xác thực Google Cloud và Clone Code"),
    
    nbf.v4.new_code_cell("""from google.colab import auth
import os

# 1. Xác thực tài khoản Google để truy cập bucket
auth.authenticate_user()

# 2. Clone source code
!git clone https://github.com/vtnguyen04/Real-Estate-Recomendation-System.git
%cd Real-Estate-Recomendation-System

# 3. Cài đặt các thư viện cần thiết
!pip install -r requirements.txt
"""),

    nbf.v4.new_markdown_cell("## Bước 2: Import Modules & Khởi tạo Môi trường"),
    
    nbf.v4.new_code_cell("""import sys
import polars as pl
import pyarrow.dataset as ds
from src.utils.logging import get_logger

logger = get_logger("colab_runner")

# Khởi tạo đường dẫn GCS
BUCKET_NAME = "datathon_2026_final"
TRAIN_PATH = f"gs://{BUCKET_NAME}/train/"
TEST_PATH = f"gs://{BUCKET_NAME}/test/"

print("Sẵn sàng dữ liệu!")
"""),

    nbf.v4.new_markdown_cell("## Bước 3: Load Data (Lazy Load để chống tràn RAM)"),
    
    nbf.v4.new_code_cell("""# Dùng pyarrow dataset để quét thư mục mà không load hết vào RAM
try:
    print("Loading test_users...")
    df_test = pl.read_parquet(f"{TEST_PATH}test_users.parquet")
    print(f"Test users: {len(df_test)}")
    
    print("Scanning dim_listing...")
    catalog_ds = ds.dataset(f"{TRAIN_PATH}dim_listing/", format="parquet")
    df_items = pl.scan_pyarrow_dataset(catalog_ds).collect()
    print(f"Items catalog: {len(df_items)}")
    
except Exception as e:
    print(f"Lỗi truy cập dữ liệu (có thể chưa auth hoặc bucket không tồn tại): {e}")
"""),

    nbf.v4.new_markdown_cell("## Bước 4: Chạy Pipeline Trực Tiếp thông qua Script\nCách nhanh nhất là gọi script `run_kaggle_submission.py` mà chúng ta đã đóng gói sẵn."),
    
    nbf.v4.new_code_cell("""# Lệnh này sẽ chạy toàn bộ quy trình: Forensics -> Features -> Models -> Reranking
!python scripts/run_kaggle_submission.py --bucket datathon_2026_final

# Khi chạy xong, file submission.csv sẽ nằm trong thư mục hiện tại.
"""),

    nbf.v4.new_markdown_cell("## Bước 5 (Tùy chọn): Chạy Debug từng bước bằng tay\nNếu bạn muốn tuỳ chỉnh logic, bạn có thể gọi thẳng các module từ `src/`."),
    
    nbf.v4.new_code_cell("""from src.pipeline.data_forensics import DataForensics
from src.features.feature_engineer import FeatureEngineer
from src.rules.geo_rules import GeoProximityScoreRule
from src.rules.quality_rules import QualityScoreRule

# Khởi tạo
forensics = DataForensics()
rules = [GeoProximityScoreRule(), QualityScoreRule()]
fe = FeatureEngineer(deterministic_rules=rules)

print("Modules đã sẵn sàng để debug!")
""")
]

nb['cells'] = cells

with open('Colab_Runner.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Created Colab_Runner.ipynb")

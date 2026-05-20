import sys, os
from datetime import timedelta
from collections import defaultdict
import polars as pl
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

from config.settings import PipelineConfig
from src.evaluation.metrics import recall_at_k
from src.utils.logging import get_logger
from src.core.base import RecommendationContext

logger = get_logger("round_18")
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..", ".cache")

def evaluate_recommender(name, recommender_func, val_users, gt, k=200):
    recalls = []
    logger.info(f"Evaluating {name}...")
    for i, u in enumerate(val_users):
        try:
            recs = recommender_func(u, k)
            recalls.append(recall_at_k(recs, gt[u], k=k))
        except Exception as e:
            logger.error(f"{name} failed for user {u}: {e}")
            break
        
        if (i+1) % 1000 == 0:
            logger.info(f"  {name}: {i+1}/{len(val_users)} users evaluated")
    
    score = np.mean(recalls) if recalls else 0.0
    logger.info(f"--> {name} Recall@{k}: {score:.4f}")
    return score

def main():
    config = PipelineConfig()
    logger.info("ROUND 18: INDEPENDENT CANDIDATE GENERATOR EVALUATION")
    
    contacts = pl.read_parquet(os.path.join(CACHE_DIR, "contact_pairs.parquet"))
    date_range = pl.read_parquet(os.path.join(CACHE_DIR, "date_range.parquet"))
    max_date = date_range["max_date"][0]
    split_date = max_date - timedelta(days=config.validation_days)
    
    train_contacts = contacts.filter(pl.col("last_date") <= split_date)
    val_contacts = contacts.filter(pl.col("last_date") > split_date)
    
    gt = defaultdict(set)
    for r in val_contacts.iter_rows(named=True):
        gt[r["user_id"]].add(r["item_id"])
        
    all_val_users = list(gt.keys())
    # Sample 2,000 users for quick but representative evaluation
    rng = np.random.default_rng(42)
    val_users = rng.choice(all_val_users, size=min(2000, len(all_val_users)), replace=False).tolist()
    user_set = set(val_users)
    logger.info(f"Evaluating on {len(val_users)} users")

    df_listing_path = os.path.join(config.data.train_path, "dim_listing")
    if os.path.isdir(df_listing_path):
        df_listing = pl.scan_parquet(os.path.join(df_listing_path, "*.parquet")).collect()
    else:
        df_listing = pl.read_parquet(df_listing_path + ".parquet")

    results = {}

    # 1. Pageview Replay
    from src.models.candidates.pageview_replay import PageviewReplayRecommender
    pv = PageviewReplayRecommender(window_days=14, max_items_per_user=200)
    events_path = os.path.join(config.data.train_path, "fact_user_events/*.parquet")
    pv.fit(events_path, user_ids=user_set, cutoff_date=split_date)
    
    results["1. PageviewReplay"] = evaluate_recommender(
        "PageviewReplay", 
        lambda u, k: pv.recommend(u, k=k), 
        val_users, gt
    )

    # 2. LightALS
    from src.models.candidates.light_als import LightALSRecommender
    als = LightALSRecommender()
    als.load(os.path.join("outputs/models/", "als"))
    if als._matrix is None:
        als_contacts = pl.read_parquet(os.path.join(CACHE_DIR, "als_contact_pairs.parquet"))
        als.rebuild_matrix(als_contacts)
    als_batch = als.recommend_batch(val_users, n=200, return_scores=False)
    
    results["2. LightALS"] = evaluate_recommender(
        "LightALS", 
        lambda u, k: als_batch.get(u, []), 
        val_users, gt
    )

    # 3. IntentRecommender
    from src.models.candidates.intent_recommender import IntentRecommender
    intent_rec = IntentRecommender(max_items_per_intent=200)
    pvs_lazy = pl.scan_parquet(events_path).filter(
        (pl.col("event_ts") <= split_date) & 
        (pl.col("event_ts") >= split_date - pl.duration(days=14)) &
        (pl.col("event_type") == "pageview")
    ).select(["user_id", "item_id"]).collect()
    valid_items = set(df_listing["item_id"].to_list())
    intent_rec.fit(pvs=pvs_lazy, dim_listing=df_listing, valid_items=valid_items)

    results["3. IntentRecommender"] = evaluate_recommender(
        "IntentRecommender", 
        lambda u, k: intent_rec.recommend(u, k=k, exclude=set()), 
        val_users, gt
    )

    # 4. UserKNN
    try:
        from src.models.candidates.user_knn import UserKNNRecommender
        user_knn = UserKNNRecommender(max_neighbors_per_item=30)
        # Using lazy frame for train_data
        train_lazy = train_contacts.lazy()
        user_knn.fit(train_lazy, query_user_ids=user_set, valid_items=valid_items)
        
        def knn_rec(u, k):
            ctx = RecommendationContext(user_id=u, num_recommendations=k)
            df = user_knn.recommend(ctx).collect()
            return df["item_id"].to_list() if "item_id" in df.columns else []

        results["4. UserKNN"] = evaluate_recommender("UserKNN", knn_rec, val_users, gt)
    except Exception as e:
        logger.error(f"UserKNN initialization failed: {e}")

    # 5. SellerExpansion
    try:
        from src.models.candidates.seller_recommender import SellerExpansionRecommender
        seller_rec = SellerExpansionRecommender(max_items_per_seller=50)
        seller_rec.fit(train_lazy, listing_df=df_listing, query_user_ids=user_set)
        
        def seller_rec_fn(u, k):
            ctx = RecommendationContext(user_id=u, num_recommendations=k)
            df = seller_rec.recommend(ctx).collect()
            return df["item_id"].to_list() if "item_id" in df.columns else []

        results["5. SellerExpansion"] = evaluate_recommender("SellerExpansion", seller_rec_fn, val_users, gt)
    except Exception as e:
        logger.error(f"SellerExpansion initialization failed: {e}")

    # 6. Segment Popularity
    from src.models.candidates.segment_popularity import SegmentPopularityRecommender
    segpop = SegmentPopularityRecommender().load(os.path.join("outputs/models/", "segpop.pkl"))
    # Extract prefs for val users
    prefs_df = (
        train_contacts.filter(pl.col("user_id").is_in(val_users))
        .group_by("user_id")
        .agg([
            pl.col("city_name").drop_nulls().mode().first().alias("pref_city"),
            pl.col("category").drop_nulls().cast(pl.Int64).mode().first().alias("pref_cat"),
        ])
    )
    prefs_dict = {r["user_id"]: (r.get("pref_city"), r.get("pref_cat")) for r in prefs_df.iter_rows(named=True)}
    
    def segpop_rec_fn(u, k):
        city, cat = prefs_dict.get(u, (None, None))
        return segpop.get_segment_items(pref_city=city, pref_cat=cat, k=k)

    results["6. SegmentPopularity"] = evaluate_recommender("SegmentPopularity", segpop_rec_fn, val_users, gt)

    # Print summary
    logger.info("========================================")
    logger.info("CANDIDATE GENERATOR EVALUATION (Recall@200)")
    logger.info("========================================")
    
    report_path = "src/eda/reports/round_18_report.md"
    with open(report_path, "w") as f:
        f.write("# Round 18 Report: Independent Candidate Generator Evaluation\n\n")
        f.write("## Executive Summary\n")
        f.write("Đánh giá công bằng mức trần Recall@200 của TẤT CẢ các mô hình Candidate Generation đang có trong `src/models/candidates/` để dẹp bỏ bias.\n\n")
        f.write("## Data Evidence\n")
        f.write("```\n")
        for name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"{name:20}: {score:.4f}")
            f.write(f"{name:25}: {score:.4f}\n")
        f.write("```\n\n")
        
        f.write("## Domain Explanation & Next Steps\n")
        f.write("Kết quả này sẽ là kim chỉ nam tuyệt đối để sắp xếp Priority trong CascadeCandidateGenerator.\n")
    
    logger.info(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()

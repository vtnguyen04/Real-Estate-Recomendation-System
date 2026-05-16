import logging
import warnings
import numpy as np
import polars as pl
from pathlib import Path

try:
    from skopt import gp_minimize
    from skopt.space import Real
    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False

from src.models.rerankers.multi_objective import MultiObjectiveReranker
from src.core.base import RecommendationContext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

def mock_get_candidates(user_id: str, k: int = 30):
    """
    Mock function to simulate getting top-k candidates from ensemble.
    In real usage, this connects to WeightedEnsembleRecommender.
    """
    return pl.DataFrame({
        "item_id": [f"item_{i}" for i in range(k)],
        "score": np.random.rand(k),
        "ml_score": np.random.rand(k),  # Normalized [0, 1] accuracy score
        "category": np.random.choice([1010, 1020, 1030, 1040, 1050], size=k),
        "seller_type": np.random.choice(["agent", "private"], size=k),
        "listing_age_days": np.random.randint(0, 100, size=k),
        "city_name": np.random.choice(["HCM", "HN", "DN"], size=k),
        "district_name": np.random.choice(["D1", "D2", "D3"], size=k),
        "ward_name": np.random.choice(["W1", "W2"], size=k),
        "price_bucket": np.random.choice(["low", "medium", "high"], size=k),
        "bedrooms": np.random.choice([1, 2, 3, None], size=k)
    }).lazy()

def compute_recall_at_k(recommended_items, ground_truth_items, k=10):
    if len(ground_truth_items) == 0:
        return 0.0
    top_k = recommended_items[:k]
    hits = sum(1 for item in top_k if item in ground_truth_items)
    return hits / min(len(ground_truth_items), k)

def main():
    if not HAS_SKOPT:
        logger.error("skopt is required for hyperparameter tuning. Run: uv pip install scikit-optimize")
        return

    logger.info("Starting Multi-Objective Weight Tuning (Bayesian Optimization)...")

    # Simulate validation users and ground truth
    val_users = [f"user_{i}" for i in range(100)] # Smaller subset for speed
    
    # Pre-configure ground truth data
    ground_truth = {user: [f"item_{np.random.randint(0, 30)}"] for user in val_users}
    
    gt_distributions = {
        'agent_ratio': 0.65,
        'category_dist': {1010: 0.3, 1020: 0.4, 1030: 0.1, 1040: 0.1, 1050: 0.1}
    }

    def objective(params):
        beta, gamma, delta = params
        alpha = 1.0 - beta - gamma - delta

        # Constraint: accuracy must dominate
        if alpha < 0.5:
            return 1.0  # Return bad score (we want to minimize negative composite)

        reranker = MultiObjectiveReranker(alpha, beta, gamma, delta, gt_distributions)

        recalls, diversities, fairnesses, freshnesses = [], [], [], []

        for user_id in val_users:
            # 1. Get 30 candidates from ensemble
            candidates_lazy = mock_get_candidates(user_id, k=30)
            
            # 2. Re-rank to get top 10
            context = RecommendationContext(user_id=user_id, num_recommendations=10)
            final_recs_df = reranker.recommend(context, candidates_lazy).collect()
            
            if final_recs_df.is_empty():
                continue
                
            # Compute metrics
            final_items = final_recs_df['item_id'].to_list()
            gt_items = ground_truth.get(user_id, [])
            
            recall = compute_recall_at_k(final_items, gt_items, k=10)
            
            # Use reranker's metrics module directly for calculation
            # Converting to list of dicts/structs to mimic item objects
            item_dicts = final_recs_df.to_dicts()
            
            class ItemWrapper:
                def __init__(self, d):
                    for k, v in d.items():
                        setattr(self, k, v)
                        
            item_objs = [ItemWrapper(d) for d in item_dicts]
            
            diversity = reranker.metrics.compute_diversity(item_objs)
            fairness = reranker.metrics.compute_fairness(item_objs)
            freshness = reranker.metrics.compute_freshness(item_objs)

            recalls.append(recall)
            diversities.append(diversity)
            fairnesses.append(fairness)
            freshnesses.append(freshness)

        if not recalls:
            return 1.0

        # Composite metric to maximize (weighted sum of objectives)
        composite = (
            0.6 * np.mean(recalls) +
            0.2 * np.mean(diversities) +
            0.2 * np.mean(fairnesses)
        )

        return -composite  # Minimize negative composite

    # Define search space for Bayesian Optimization
    space = [
        Real(0.05, 0.25, name='beta'),
        Real(0.05, 0.25, name='gamma'),
        Real(0.03, 0.15, name='delta')
    ]

    logger.info("Running gp_minimize...")
    result = gp_minimize(
        objective,
        space,
        n_calls=20,  # Reduced for demonstration
        random_state=42,
        verbose=True
    )

    best_beta, best_gamma, best_delta = result.x
    best_alpha = 1.0 - sum(result.x)

    logger.info("=== Optimal Multi-Objective Weights Found ===")
    logger.info(f"α (Accuracy) = {best_alpha:.3f}")
    logger.info(f"β (Diversity) = {best_beta:.3f}")
    logger.info(f"γ (Fairness) = {best_gamma:.3f}")
    logger.info(f"δ (Freshness) = {best_delta:.3f}")
    
    # Ablation Study Simulation
    logger.info("Running Ablation Study comparisons...")
    configs = [
        {'name': 'Pure ML', 'alpha': 1.0, 'beta': 0.0, 'gamma': 0.0, 'delta': 0.0},
        {'name': 'High Accuracy', 'alpha': 0.8, 'beta': 0.1, 'gamma': 0.05, 'delta': 0.05},
        {'name': 'Balanced (Ours)', 'alpha': best_alpha, 'beta': best_beta, 'gamma': best_gamma, 'delta': best_delta},
        {'name': 'High Health', 'alpha': 0.5, 'beta': 0.2, 'gamma': 0.2, 'delta': 0.1},
    ]

    results = []
    for config in configs:
        name = config.pop('name')
        # Here we mock the evaluation, but you would normally run full evaluation
        # Similar loop as objective()
        results.append({
            'Config': name,
            'Recall@10': 0.35 * config['alpha'], # mock value
            'Diversity': 0.60 * (config['beta'] * 5 + 0.1),
            'Fairness': 0.65 * (config['gamma'] * 4 + 0.2),
            'Freshness': 0.70 * (config['delta'] * 8 + 0.3)
        })

    df_ablation = pl.DataFrame(results)
    print(df_ablation)
    
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    df_ablation.write_csv(out_dir / "ablation_study.csv")
    logger.info(f"Ablation study saved to {out_dir / 'ablation_study.csv'}")

if __name__ == "__main__":
    main()

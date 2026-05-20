import numpy as np
import polars as pl
from typing import List, Dict, Any
from collections import Counter
from scipy.stats import entropy

class HealthMetrics:
    """
    Advanced Health Metrics for Real Estate Recommendation System.
    Includes:
    - Intra-List Diversity (ILD)
    - Category Entropy
    - Fairness (Seller Type & Category distribution match)
    - Freshness (Exponential decay)

    gt_dist is calibrated from actual training contact data by
    src/eda/round_09_health_baseline.py and stored in .cache/gt_dist.json.
    """
    # Default calibrated from EDA R09 (training contact distribution)
    _DEFAULT_GT_DIST: Dict[str, Any] = {
        "agent_ratio": 0.520,
        "category_dist": {
            1010: 0.156, 1020: 0.446, 1030: 0.065, 1040: 0.102, 1050: 0.231
        },
    }

    def __init__(self, ground_truth_dist: Dict[str, Any] = None, gt_dist_path: str = None):
        """
        Args:
            ground_truth_dist: Explicit gt_dist dict (overrides file if provided).
            gt_dist_path:      Path to JSON file produced by EDA R09.
                               Falls back to _DEFAULT_GT_DIST if file not found.
        """
        if ground_truth_dist is not None:
            self.gt_dist = ground_truth_dist
        elif gt_dist_path is not None:
            import json, os
            try:
                with open(gt_dist_path) as f:
                    raw = json.load(f)
                # JSON keys are strings; coerce category keys to int
                raw["category_dist"] = {int(k): v for k, v in raw["category_dist"].items()}
                self.gt_dist = raw
            except Exception:
                self.gt_dist = self._DEFAULT_GT_DIST
        else:
            self.gt_dist = self._DEFAULT_GT_DIST

    def compute_diversity(self, items: List[Dict[str, Any]]) -> float:
        """
        ILD based on category and location.
        """
        if len(items) <= 1:
            return 1.0
            
        def item_sim(i1, i2):
            sim = 0.0
            if i1.get('category') == i2.get('category'): sim += 0.4
            if i1.get('city_name') == i2.get('city_name'):
                sim += 0.3
                if i1.get('district_name') == i2.get('district_name'): sim += 0.3
            return sim

        n = len(items)
        total_dissim = 0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_dissim += (1.0 - item_sim(items[i], items[j]))
                count += 1
        
        ild = total_dissim / count if count > 0 else 0
        
        # Category entropy
        cats = [it.get('category') for it in items]
        counts = Counter(cats)
        probs = np.array(list(counts.values())) / n
        ent = entropy(probs, base=2)
        max_ent = np.log2(len(counts)) if len(counts) > 1 else 1
        norm_ent = ent / max_ent if max_ent > 0 else 0
        
        return 0.6 * ild + 0.4 * norm_ent

    def compute_fairness(self, items: List[Dict[str, Any]]) -> float:
        """
        KL Divergence from Ground Truth for Seller Type and Category.
        """
        if not items: return 0.0
        n = len(items)
        
        # Seller type fairness
        agent_ratio = sum([1 for it in items if it.get('seller_type') == 'agent']) / n
        seller_fairness = 1.0 - abs(agent_ratio - self.gt_dist['agent_ratio'])
        
        # Category fairness (KL divergence)
        cat_counts = Counter([it.get('category') for it in items])
        observed = np.array([cat_counts.get(c, 0) / n for c in self.gt_dist['category_dist'].keys()])
        expected = np.array([self.gt_dist['category_dist'][c] for c in self.gt_dist['category_dist'].keys()])
        
        # Small epsilon to avoid log(0)
        eps = 1e-10
        kl_div = np.sum(observed * np.log((observed + eps) / (expected + eps)))
        cat_fairness = np.exp(-kl_div) # Higher is better
        
        return 0.6 * seller_fairness + 0.4 * cat_fairness

    def compute_freshness(self, items: List[Dict[str, Any]]) -> float:
        """
        Exponential decay based on listing age.
        """
        if not items: return 0.0
        # Score = exp(-0.05 * age_days)
        scores = [np.exp(-0.05 * it.get('listing_age_days', 30)) for it in items]
        return float(np.mean(scores))

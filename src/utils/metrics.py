import numpy as np
from typing import List, Set, Dict, Any

def compute_precision_at_k(actual: List[str], recommended: List[str], k: int) -> float:
    """
    Precision@K: Proportion of recommended items in the top-K that are relevant.
    """
    if not recommended or k <= 0:
        return 0.0
    recommended_k = recommended[:k]
    actual_set = set(actual)
    relevant_in_top_k = [item for item in recommended_k if item in actual_set]
    return len(relevant_in_top_k) / k

def compute_recall_at_k(actual: List[str], recommended: List[str], k: int) -> float:
    """
    Recall@K: Proportion of relevant items that are found in the top-K recommendations.
    """
    if not actual or k <= 0:
        return 0.0
    recommended_k = recommended[:k]
    actual_set = set(actual)
    relevant_in_top_k = [item for item in recommended_k if item in actual_set]
    return len(relevant_in_top_k) / len(actual)

def compute_ndcg_at_k(actual: List[str], recommended: List[str], k: int) -> float:
    """
    NDCG@K: Normalized Discounted Cumulative Gain at K.
    Accounts for the rank of relevant items.
    """
    if not actual or not recommended or k <= 0:
        return 0.0
    
    actual_set = set(actual)
    recommended_k = recommended[:k]
    
    dcg = 0.0
    for i, item in enumerate(recommended_k):
        if item in actual_set:
            dcg += 1.0 / np.log2(i + 2)
            
    idcg = 0.0
    for i in range(min(len(actual), k)):
        idcg += 1.0 / np.log2(i + 2)
        
    return dcg / idcg if idcg > 0 else 0.0

def compute_map(actual: List[str], recommended: List[str]) -> float:
    """
    Mean Average Precision: Average of Precision@K at each rank where a relevant item is found.
    """
    if not actual or not recommended:
        return 0.0
    
    actual_set = set(actual)
    score = 0.0
    num_hits = 0
    for i, item in enumerate(recommended):
        if item in actual_set:
            num_hits += 1
            score += num_hits / (i + 1)
            
    return score / len(actual)

def compute_diversity(recommended: List[str], item_metadata: Dict[str, Dict[str, Any]], feature: str) -> float:
    """
    Intra-list Diversity: Measures how diverse the recommended items are based on a categorical feature.
    Calculated as (1 - SIM), where SIM is the proportion of pairs sharing the same feature value.
    """
    if not recommended or len(recommended) < 2:
        return 1.0
        
    values = [item_metadata.get(item_id, {}).get(feature) for item_id in recommended]
    values = [v for v in values if v is not None]
    
    if len(values) < 2:
        return 1.0
        
    matches = 0
    total_pairs = 0
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            total_pairs += 1
            if values[i] == values[j]:
                matches += 1
                
    return 1.0 - (matches / total_pairs)

def compute_novelty(recommended: List[str], item_popularity: Dict[str, int], total_users: int) -> float:
    """
    Novelty: Measures how 'unexpected' the recommendations are.
    Calculated as the mean self-information of the recommended items: -log2(p(i)).
    """
    if not recommended:
        return 0.0
        
    novelty_scores = []
    for item_id in recommended:
        # p(i) is the probability of an item being interacted with
        pop = item_popularity.get(item_id, 0)
        p_i = (pop + 1) / (total_users + 1)
        novelty_scores.append(-np.log2(p_i))
        
    return float(np.mean(novelty_scores))

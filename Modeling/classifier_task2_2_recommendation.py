
"""
XGBoost Task 2: Paper Recommendation System (Silent Mode)
All results saved to JSON, minimal console output
"""

import numpy as np
import json
import joblib
import os
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import xgboost as xgb
from datetime import datetime

# 1. Configuration


CONFIG = {
    'models_dir': 'models',
    'data_dir': '../processed_data/TIBKAT',
    'output_file': 'models/paper_recommendation_results.json',
    
    'evaluation': {
        'use_sampling': True,
        'sample_size': 500,
        'random_seed': 42
    },
    
    'recommendation': {
        'K_values': [5, 10, 20, 30, 50],
        'label_threshold': 0.15,
        'min_common_labels': 1,
        'max_candidates': 1000,
        'use_task1_params': True
    },
    
    'similarity': {
        'alpha': 0.6
    }
}


# 2. Helper Functions


def enhance_features(X, scaler):
    X_scaled = scaler.transform(X)
    X_mean = X.mean(axis=1, keepdims=True)
    X_std = X.std(axis=1, keepdims=True)
    X_max = X.max(axis=1, keepdims=True)
    X_min = X.min(axis=1, keepdims=True)
    n_squared_features = min(100, X.shape[1])
    X_squared = X[:, :n_squared_features] ** 2
    X_enhanced = np.hstack([X_scaled, X_mean, X_std, X_max, X_min, X_squared])
    return X_enhanced


def calculate_label_similarity(labels1, labels2):
    norm1 = np.linalg.norm(labels1)
    norm2 = np.linalg.norm(labels2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(labels1, labels2) / (norm1 * norm2))


def hybrid_similarity(query_embedding, query_labels, 
                     candidate_embeddings, candidate_labels,
                     alpha=0.6):
    emb_similarities = cosine_similarity(
        query_embedding.reshape(1, -1),
        candidate_embeddings
    )[0]
    
    label_similarities = np.array([
        calculate_label_similarity(query_labels, cand_labels)
        for cand_labels in candidate_labels
    ])
    
    hybrid_sim = alpha * label_similarities + (1 - alpha) * emb_similarities
    return hybrid_sim


def recommend_papers_hybrid(query_idx, query_embedding, query_pred_labels,
                            train_embeddings, train_pred_labels, train_ids,
                            K=10, label_threshold=0.15, min_common_labels=1,
                            max_candidates=1000, alpha=0.6):
    query_labels_binary = (query_pred_labels >= label_threshold).astype(int)
    train_labels_binary = (train_pred_labels >= label_threshold).astype(int)
    
    common_label_counts = np.sum(
        query_labels_binary * train_labels_binary,
        axis=1
    )
    
    candidate_mask = common_label_counts >= min_common_labels
    candidate_indices = np.where(candidate_mask)[0]
    
    if len(candidate_indices) > max_candidates:
        top_common_indices = np.argsort(common_label_counts)[-max_candidates:]
        candidate_indices = top_common_indices
    
    if len(candidate_indices) < K:
        emb_similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            train_embeddings
        )[0]
        top_k_indices = np.argsort(emb_similarities)[-K:][::-1]
        return top_k_indices, emb_similarities[top_k_indices]
    
    candidate_embeddings = train_embeddings[candidate_indices]
    candidate_labels = train_pred_labels[candidate_indices]
    
    similarities = hybrid_similarity(
        query_embedding,
        query_pred_labels,
        candidate_embeddings,
        candidate_labels,
        alpha=alpha
    )
    
    if len(similarities) >= K:
        top_k_in_candidates = np.argsort(similarities)[-K:][::-1]
        recommended_indices = candidate_indices[top_k_in_candidates]
        recommended_similarities = similarities[top_k_in_candidates]
    else:
        sorted_indices = np.argsort(similarities)[::-1]
        recommended_indices = candidate_indices[sorted_indices]
        recommended_similarities = similarities[sorted_indices]
    
    return recommended_indices, recommended_similarities


def evaluate_recommendations(query_true_labels, recommended_true_labels, K):
    relevance_scores = []
    for rec_labels in recommended_true_labels[:K]:
        common_count = np.sum(query_true_labels * rec_labels)
        relevance_scores.append(common_count)
    
    relevance_scores = np.array(relevance_scores)
    is_relevant = (relevance_scores > 0).astype(int)
    
    precision_at_k = is_relevant.sum() / K if K > 0 else 0
    recall_at_k = precision_at_k
    
    if precision_at_k + recall_at_k > 0:
        f1_at_k = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)
    else:
        f1_at_k = 0
    
    dcg = np.sum(relevance_scores / np.log2(np.arange(2, K + 2)))
    ideal_relevance = np.sort(relevance_scores)[::-1]
    idcg = np.sum(ideal_relevance / np.log2(np.arange(2, K + 2)))
    ndcg = dcg / idcg if idcg > 0 else 0
    
    hit_rate = 1 if is_relevant.sum() > 0 else 0
    
    return {
        'precision_at_k': float(precision_at_k),
        'recall_at_k': float(recall_at_k),
        'f1_at_k': float(f1_at_k),
        'ndcg_at_k': float(ndcg),
        'hit_rate': int(hit_rate),
        'n_relevant': int(is_relevant.sum()),
        'avg_relevance': float(relevance_scores.mean())
    }


# 3. Load Models and Data



models_dir = CONFIG['models_dir']
data_dir = CONFIG['data_dir']

xgb_models = joblib.load(os.path.join(models_dir, 'xgboost_models_optimized.pkl'))
mlb = joblib.load(os.path.join(models_dir, 'label_binarizer.pkl'))
feature_scaler = joblib.load(os.path.join(models_dir, 'feature_scaler.pkl'))
calibrators = joblib.load(os.path.join(models_dir, 'probability_calibrators.pkl'))

task1_results_path = os.path.join(models_dir, 'optimized_results.json')
if CONFIG['recommendation']['use_task1_params'] and os.path.exists(task1_results_path):
    with open(task1_results_path, 'r') as f:
        task1_results = json.load(f)
    CONFIG['recommendation']['label_threshold'] = task1_results['optimization']['best_params']['threshold']



# 4. Load Train Set

X_train = np.load(os.path.join(data_dir, 'embedding/tibkat_train_embeddings.npy'))
train_ids = json.load(open(os.path.join(data_dir, 'embedding/tibkat_train_ids.json')))

train_labels_dict = {}
with open(os.path.join(data_dir, 'translating/train_all.jsonl'), 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        train_labels_dict[data['paper_id']] = data.get('subject', {}).get('labels', [])

y_train_raw = [train_labels_dict.get(pid, []) for pid in train_ids]
y_train_true = mlb.transform(y_train_raw)

X_train_enhanced = enhance_features(X_train, feature_scaler)
dmatrix_train = xgb.DMatrix(X_train_enhanced)

y_train_pred_scores = np.zeros((len(X_train_enhanced), len(xgb_models)))

for i in tqdm(range(len(xgb_models)), desc="Train pred", ncols=80, leave=False):
    if xgb_models[i] is not None:
        y_train_pred_scores[:, i] = xgb_models[i].predict(dmatrix_train)

for i, calibrator in enumerate(calibrators):
    if calibrator is not None:
        y_train_pred_scores[:, i] = calibrator.transform(y_train_pred_scores[:, i])


# 5. Load Test Set


X_test = np.load(os.path.join(data_dir, 'embedding/tibkat_test_embeddings.npy'))
test_ids = json.load(open(os.path.join(data_dir, 'embedding/tibkat_test_ids.json')))

test_labels_dict = {}
with open(os.path.join(data_dir, 'translating/test_all.jsonl'), 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        test_labels_dict[data['paper_id']] = data.get('subject', {}).get('labels', [])

y_test_raw = [test_labels_dict.get(pid, []) for pid in test_ids]
y_test_true = mlb.transform(y_test_raw)


X_test_enhanced = enhance_features(X_test, feature_scaler)
dmatrix_test = xgb.DMatrix(X_test_enhanced)

y_test_pred_scores = np.zeros((len(X_test_enhanced), len(xgb_models)))

for i in tqdm(range(len(xgb_models)), desc="Test pred", ncols=80, leave=False):
    if xgb_models[i] is not None:
        y_test_pred_scores[:, i] = xgb_models[i].predict(dmatrix_test)

for i, calibrator in enumerate(calibrators):
    if calibrator is not None:
        y_test_pred_scores[:, i] = calibrator.transform(y_test_pred_scores[:, i])



# 6. Sampling


if CONFIG['evaluation']['use_sampling']:
    np.random.seed(CONFIG['evaluation']['random_seed'])
    sample_size = min(CONFIG['evaluation']['sample_size'], len(X_test))
    eval_indices = np.random.choice(len(X_test), size=sample_size, replace=False)
    eval_indices = np.sort(eval_indices)
    print(f" Sampled {len(eval_indices)}/{len(X_test)} papers")
else:
    eval_indices = np.arange(len(X_test))
    print(f" Using all {len(eval_indices)} papers")


# 7. Paper Recommendation



K_values = CONFIG['recommendation']['K_values']
results_by_K = {}

for K in K_values:
    all_metrics = []
    
    for i in tqdm(eval_indices, desc=f"K={K}", ncols=80, leave=False):
        recommended_indices, similarities = recommend_papers_hybrid(
            query_idx=i,
            query_embedding=X_test[i],
            query_pred_labels=y_test_pred_scores[i],
            train_embeddings=X_train,
            train_pred_labels=y_train_pred_scores,
            train_ids=train_ids,
            K=K,
            label_threshold=CONFIG['recommendation']['label_threshold'],
            min_common_labels=CONFIG['recommendation']['min_common_labels'],
            max_candidates=CONFIG['recommendation']['max_candidates'],
            alpha=CONFIG['similarity']['alpha']
        )
        
        query_true_labels = y_test_true[i]
        recommended_true_labels = y_train_true[recommended_indices]
        
        metrics = evaluate_recommendations(
            query_true_labels,
            recommended_true_labels,
            K
        )
        
        all_metrics.append(metrics)
    
    results_by_K[K] = {
        'K': K,
        'precision_at_k': float(np.mean([m['precision_at_k'] for m in all_metrics])),
        'recall_at_k': float(np.mean([m['recall_at_k'] for m in all_metrics])),
        'f1_at_k': float(np.mean([m['f1_at_k'] for m in all_metrics])),
        'ndcg_at_k': float(np.mean([m['ndcg_at_k'] for m in all_metrics])),
        'hit_rate': float(np.mean([m['hit_rate'] for m in all_metrics])),
        'avg_n_relevant': float(np.mean([m['n_relevant'] for m in all_metrics])),
        'n_queries': len(eval_indices)
    }
    

# 8. Summary Statistics


best_k_hit = max(results_by_K.items(), key=lambda x: x[1]['hit_rate'])[0]
best_k_prec = max(results_by_K.items(), key=lambda x: x[1]['precision_at_k'])[0]
best_k_ndcg = max(results_by_K.items(), key=lambda x: x[1]['ndcg_at_k'])[0]

summary_stats = {
    'best_K_for_hit_rate': best_k_hit,
    'best_K_for_precision': best_k_prec,
    'best_K_for_ndcg': best_k_ndcg,
    'avg_precision_across_K': float(np.mean([r['precision_at_k'] for r in results_by_K.values()])),
    'avg_hit_rate_across_K': float(np.mean([r['hit_rate'] for r in results_by_K.values()])),
    'avg_ndcg_across_K': float(np.mean([r['ndcg_at_k'] for r in results_by_K.values()]))
}


# 9. Save Results


output_data = {
    'metadata': {
        'task': 'paper_recommendation',
        'description': 'Hybrid similarity-based paper recommendation',
        'timestamp': datetime.now().isoformat(),
        'n_train_papers': len(X_train),
        'n_test_papers': len(X_test),
        'n_evaluated_papers': len(eval_indices),
        'sampling_used': CONFIG['evaluation']['use_sampling'],
        'sample_size': len(eval_indices),
        'sample_ratio': float(len(eval_indices) / len(X_test)),
        'n_labels': len(mlb.classes_),
        'statistical_confidence_95ci': f"±{1.96 / np.sqrt(len(eval_indices)) * 100:.1f}%"
    },
    
    'configuration': {
        'evaluation': CONFIG['evaluation'],
        'recommendation': CONFIG['recommendation'],
        'similarity': CONFIG['similarity']
    },
    
    'results': {
        'by_K': results_by_K,
        'summary': summary_stats
    },
    
    'interpretation': {
        K: {
            'description': f"When recommending {K} papers per query",
            'precision': f"{results_by_K[K]['precision_at_k']*100:.1f}% of recommended papers are relevant",
            'hit_rate': f"{results_by_K[K]['hit_rate']*100:.1f}% of queries get at least 1 relevant paper",
            'ndcg': f"Ranking quality score: {results_by_K[K]['ndcg_at_k']:.4f}",
            'avg_relevant': f"Average {results_by_K[K]['avg_n_relevant']:.2f} relevant papers in top-{K}",
            'evaluated_on': f"{len(eval_indices)} {'sampled' if CONFIG['evaluation']['use_sampling'] else 'total'} papers"
        }
        for K in K_values
    },
    
    'quality_assessment': {
        'overall_rating': ' Excellent' if summary_stats['avg_hit_rate_across_K'] >= 0.85 
                         else ' Good' if summary_stats['avg_hit_rate_across_K'] >= 0.75
                         else ' Acceptable' if summary_stats['avg_hit_rate_across_K'] >= 0.65
                         else 'Needs Improvement',
        'deployment_ready': summary_stats['avg_hit_rate_across_K'] >= 0.70,
        'recommended_K': best_k_hit if results_by_K[best_k_hit]['hit_rate'] >= 0.80 else 20,
        'best_performance': {
            'highest_hit_rate': {
                'K': best_k_hit,
                'value': results_by_K[best_k_hit]['hit_rate']
            },
            'highest_precision': {
                'K': best_k_prec,
                'value': results_by_K[best_k_prec]['precision_at_k']
            },
            'highest_ndcg': {
                'K': best_k_ndcg,
                'value': results_by_K[best_k_ndcg]['ndcg_at_k']
            }
        }
    },
    
    'detailed_analysis': {
        'K_comparison': {
            str(K): {
                'precision_rank': sorted(results_by_K.keys(), 
                                       key=lambda k: results_by_K[k]['precision_at_k'], 
                                       reverse=True).index(K) + 1,
                'hit_rate_rank': sorted(results_by_K.keys(), 
                                      key=lambda k: results_by_K[k]['hit_rate'], 
                                      reverse=True).index(K) + 1,
                'ndcg_rank': sorted(results_by_K.keys(), 
                                  key=lambda k: results_by_K[k]['ndcg_at_k'], 
                                  reverse=True).index(K) + 1
            }
            for K in K_values
        }
    }
}

output_path = CONFIG['output_file']
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)



#XGBoost Task 2: Recall Optimization

import numpy as np
import json
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, hamming_loss
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import xgboost as xgb
from datetime import datetime



# 1. Configuration

CONFIG = {
    'models_dir': 'models',
    'data_dir': '../processed_data/TIBKAT',
    'min_precision': 0.25,
    'output_file': 'models/recall_optimized_results.json',
    
    'max_k_values': [10, 15, 20, 30, 40, 50, 60, 80],
    'min_k_values': [1, 2, 3, 5, 7, 10],
    'threshold_values': [0.03, 0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.25, 0.3],
    
    'train_val_split': {
        'test_size': 3000,
        'random_state': 42,
        'shuffle': True
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

def predict_with_adaptive_topk_threshold(y_scores, max_k=20, min_k=1, threshold=0.15):
    n_samples = y_scores.shape[0]
    y_pred = np.zeros_like(y_scores, dtype=int)
    
    for i in range(n_samples):
        scores_i = y_scores[i]
        top_k_indices = np.argsort(scores_i)[-max_k:]
        top_k_scores = scores_i[top_k_indices]
        above_threshold = top_k_scores >= threshold
        selected_indices = top_k_indices[above_threshold]
        
        if len(selected_indices) < min_k:
            selected_indices = top_k_indices[-min_k:]
        
        y_pred[i, selected_indices] = 1
    
    return y_pred

def calculate_metrics(y_true, y_pred):
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    hamming = hamming_loss(y_true, y_pred)
    avg_predictions = y_pred.sum(axis=1).mean()
    
    return {
        'precision_micro': float(p_micro),
        'recall_micro': float(r_micro),
        'f1_micro': float(f1_micro),
        'precision_macro': float(p_macro),
        'recall_macro': float(r_macro),
        'f1_macro': float(f1_macro),
        'hamming_loss': float(hamming),
        'avg_predictions': float(avg_predictions)
    }

def calculate_topk_accuracy(y_true, y_scores, k):
    """Calculate Top-K hit rate"""
    n_samples = len(y_true)
    hits = 0
    total_hits = 0
    
    for i in range(n_samples):
        top_k_indices = set(np.argsort(y_scores[i])[-k:])
        true_indices = set(np.where(y_true[i] == 1)[0])
        n_hits = len(top_k_indices & true_indices)
        
        if n_hits > 0:
            hits += 1
        total_hits += n_hits
    
    topk_acc = hits / n_samples
    avg_hits = total_hits / n_samples
    
    return {
        'accuracy': float(topk_acc),
        'avg_hits': float(avg_hits),
        'hit_count': int(hits),
        'total_samples': int(n_samples)
    }

def grid_search_recall_focused(y_val_scores, y_val, config):
    """Grid search optimizing for recall"""
    best_recall = 0
    best_params = None
    all_results = []
    
    min_precision = config['min_precision']
    total_configs = len(config['max_k_values']) * len(config['min_k_values']) * len(config['threshold_values'])
    valid_configs = 0
    
    print(f"\nRunning grid search ({total_configs} configurations)...")
    
    for max_k in config['max_k_values']:
        for min_k in config['min_k_values']:
            if min_k > max_k:
                continue
            
            for threshold in config['threshold_values']:
                y_pred = predict_with_adaptive_topk_threshold(
                    y_val_scores, max_k=max_k, min_k=min_k, threshold=threshold
                )
                
                metrics = calculate_metrics(y_val, y_pred)
                
                p = metrics['precision_micro']
                r = metrics['recall_micro']
                
                meets_constraint = p >= min_precision
                
                result = {
                    'max_k': max_k,
                    'min_k': min_k,
                    'threshold': threshold,
                    **metrics,
                    'meets_constraint': meets_constraint
                }
                
                all_results.append(result)
                
                if meets_constraint:
                    valid_configs += 1
                
                if meets_constraint and r > best_recall:
                    best_recall = r
                    best_params = result.copy()
    
    if best_params is None:
        best_params = max(all_results, key=lambda x: x['f1_micro'])
    
    search_summary = {
        'total_configs': total_configs,
        'valid_configs': valid_configs,
        'best_recall': best_recall
    }
    
    print(f" Grid search completed: {valid_configs}/{total_configs} valid configs")
    print(f" Best recall: {best_recall:.4f}")
    
    return best_params, all_results, search_summary


# 3. Load Data and Recreate Split



data_dir = CONFIG['data_dir']

X_all = np.load(os.path.join(data_dir, 'embedding/tibkat_train_embeddings.npy'))
all_ids = json.load(open(os.path.join(data_dir, 'embedding/tibkat_train_ids.json')))

all_labels = {}
with open(os.path.join(data_dir, 'translating/train_all.jsonl'), 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        all_labels[data['paper_id']] = data.get('subject', {}).get('labels', [])

y_all_raw = [all_labels.get(pid, []) for pid in all_ids]

X_train, X_val, y_train_raw, y_val_raw, train_ids, val_ids = train_test_split(
    X_all, y_all_raw, all_ids,
    test_size=CONFIG['train_val_split']['test_size'],
    random_state=CONFIG['train_val_split']['random_state'],
    shuffle=CONFIG['train_val_split']['shuffle']
)



# 4. Load Models


models_dir = CONFIG['models_dir']

xgb_models = joblib.load(os.path.join(models_dir, 'xgboost_models_optimized.pkl'))
mlb = joblib.load(os.path.join(models_dir, 'label_binarizer.pkl'))
feature_scaler = joblib.load(os.path.join(models_dir, 'feature_scaler.pkl'))
calibrators = joblib.load(os.path.join(models_dir, 'probability_calibrators.pkl'))

y_val = mlb.transform(y_val_raw)

# 5. Validation Predictions



X_val_enhanced = enhance_features(X_val, feature_scaler)

y_val_scores = np.zeros((len(X_val_enhanced), len(xgb_models)))
dmatrix_val = xgb.DMatrix(X_val_enhanced)

for i in tqdm(range(len(xgb_models)), desc="Predicting", ncols=80):
    if xgb_models[i] is not None:
        y_val_scores[:, i] = xgb_models[i].predict(dmatrix_val)

for i, calibrator in enumerate(calibrators):
    if calibrator is not None:
        y_val_scores[:, i] = calibrator.transform(y_val_scores[:, i])


# 6. Grid Search on Validation


best_params, all_results, search_summary = grid_search_recall_focused(
    y_val_scores, y_val, CONFIG
)


# 7. Load Test Data


X_test = np.load(os.path.join(data_dir, 'embedding/tibkat_test_embeddings.npy'))
test_ids = json.load(open(os.path.join(data_dir, 'embedding/tibkat_test_ids.json')))

test_labels = {}
with open(os.path.join(data_dir, 'translating/test_all.jsonl'), 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        test_labels[data['paper_id']] = data.get('subject', {}).get('labels', [])

y_test_raw = [test_labels.get(pid, []) for pid in test_ids]
y_test = mlb.transform(y_test_raw)

print(f" Test samples: {len(X_test):,}")


# 8. Test Predictions


X_test_enhanced = enhance_features(X_test, feature_scaler)

y_test_scores = np.zeros((len(X_test_enhanced), len(xgb_models)))
dmatrix_test = xgb.DMatrix(X_test_enhanced)

for i in tqdm(range(len(xgb_models)), desc="Test prediction", ncols=80):
    if xgb_models[i] is not None:
        y_test_scores[:, i] = xgb_models[i].predict(dmatrix_test)

for i, calibrator in enumerate(calibrators):
    if calibrator is not None:
        y_test_scores[:, i] = calibrator.transform(y_test_scores[:, i])

y_test_pred = predict_with_adaptive_topk_threshold(
    y_test_scores,
    max_k=best_params['max_k'],
    min_k=best_params['min_k'],
    threshold=best_params['threshold']
)

test_metrics = calculate_metrics(y_test, y_test_pred)


# 9. Calculate Top-K Accuracy


k_values = [1, 3, 5, 10, 15, 20, 30]
topk_results = {}

for k in k_values:
    topk_results[k] = calculate_topk_accuracy(y_test, y_test_scores, k)


# 10. Compare with Task1


comparison_data = None
original_results_path = os.path.join(models_dir, 'optimized_results.json')

if os.path.exists(original_results_path):
    with open(original_results_path, 'r') as f:
        original_results = json.load(f)
    
    orig_test = original_results['test_results']['micro']
    orig_params = original_results['optimization']['best_params']
    
    comparison_data = {
        'task1_params': orig_params,
        'task1_results': {
            'precision_micro': orig_test['precision'],
            'recall_micro': orig_test['recall'],
            'f1_micro': orig_test['f1'],
            'avg_predictions': original_results['test_results']['avg_predictions']
        },
        'task2_params': {
            'max_k': best_params['max_k'],
            'min_k': best_params['min_k'],
            'threshold': best_params['threshold']
        },
        'task2_results': {
            'precision_micro': test_metrics['precision_micro'],
            'recall_micro': test_metrics['recall_micro'],
            'f1_micro': test_metrics['f1_micro'],
            'avg_predictions': test_metrics['avg_predictions']
        },
        'improvements': {
            'precision_change': test_metrics['precision_micro'] - orig_test['precision'],
            'precision_change_pct': ((test_metrics['precision_micro'] - orig_test['precision']) / orig_test['precision'] * 100) if orig_test['precision'] > 0 else 0,
            'recall_improvement': test_metrics['recall_micro'] - orig_test['recall'],
            'recall_improvement_pct': ((test_metrics['recall_micro'] - orig_test['recall']) / orig_test['recall'] * 100) if orig_test['recall'] > 0 else 0,
            'f1_change': test_metrics['f1_micro'] - orig_test['f1'],
            'f1_change_pct': ((test_metrics['f1_micro'] - orig_test['f1']) / orig_test['f1'] * 100) if orig_test['f1'] > 0 else 0
        }
    }
    
    print(f" Recall improvement: {comparison_data['improvements']['recall_improvement_pct']:+.1f}%")


# 11. Save Results



# Quality assessment
top3_acc = topk_results[3]['accuracy']
top5_acc = topk_results[5]['accuracy']

quality_assessment = {
    'top1_rating': '' if topk_results[1]['accuracy'] >= 0.5 else '' if topk_results[1]['accuracy'] >= 0.4 else '',
    'top3_rating': '' if top3_acc >= 0.7 else '' if top3_acc >= 0.6 else '',
    'top5_rating': '' if top5_acc >= 0.8 else '' if top5_acc >= 0.7 else '',
    'deployment_ready': top5_acc >= 0.70,
    'recommendation': 'Excellent' if top5_acc >= 0.8 else 'Good' if top5_acc >= 0.7 else 'Acceptable' if top5_acc >= 0.6 else 'Needs improvement'
}

output_data = {
    'metadata': {
        'task': 'recall_optimization',
        'description': 'Optimize recall while maintaining precision >= 25%',
        'timestamp': datetime.now().isoformat(),
        'min_precision_constraint': CONFIG['min_precision']
    },
    
    'configuration': {
        'search_space': {
            'max_k_values': CONFIG['max_k_values'],
            'min_k_values': CONFIG['min_k_values'],
            'threshold_values': CONFIG['threshold_values']
        },
        'data_split': {
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'split_params': CONFIG['train_val_split'],
            'first_5_val_ids': val_ids[:5]
        }
    },
    
    'grid_search': {
        'summary': search_summary,
        'best_config': {
            'max_k': best_params['max_k'],
            'min_k': best_params['min_k'],
            'threshold': best_params['threshold']
        },
        'validation_performance': {
            'precision_micro': best_params['precision_micro'],
            'recall_micro': best_params['recall_micro'],
            'f1_micro': best_params['f1_micro'],
            'precision_macro': best_params['precision_macro'],
            'recall_macro': best_params['recall_macro'],
            'f1_macro': best_params['f1_macro'],
            'hamming_loss': best_params['hamming_loss'],
            'avg_predictions': best_params['avg_predictions']
        },
        'top_10_configs': sorted(
            [r for r in all_results if r['meets_constraint']], 
            key=lambda x: x['recall_micro'], 
            reverse=True
        )[:10]
    },
    
    'test_results': {
        'micro_average': {
            'precision': test_metrics['precision_micro'],
            'recall': test_metrics['recall_micro'],
            'f1_score': test_metrics['f1_micro']
        },
        'macro_average': {
            'precision': test_metrics['precision_macro'],
            'recall': test_metrics['recall_macro'],
            'f1_score': test_metrics['f1_macro']
        },
        'other_metrics': {
            'hamming_loss': test_metrics['hamming_loss'],
            'avg_predictions': test_metrics['avg_predictions']
        }
    },
    
    'topk_accuracy': {
        'detailed_results': topk_results,
        'summary': {
            'top1_accuracy': topk_results[1]['accuracy'],
            'top3_accuracy': topk_results[3]['accuracy'],
            'top5_accuracy': topk_results[5]['accuracy'],
            'top10_accuracy': topk_results[10]['accuracy'],
            'top3_avg_hits': topk_results[3]['avg_hits'],
            'top5_avg_hits': topk_results[5]['avg_hits']
        },
        'quality_assessment': quality_assessment,
        'interpretation': {
            'top1': f"{topk_results[1]['accuracy']*100:.1f}% of papers have correct label in top-1 recommendation",
            'top3': f"{topk_results[3]['accuracy']*100:.1f}% of papers have at least 1 correct label in top-3",
            'top5': f"{topk_results[5]['accuracy']*100:.1f}% of papers have at least 1 correct label in top-5",
            'top10': f"{topk_results[10]['accuracy']*100:.1f}% of papers have at least 1 correct label in top-10"
        }
    },
    
    'comparison_with_task1': comparison_data,
    
    'summary': {
        'validation_recall': best_params['recall_micro'],
        'validation_precision': best_params['precision_micro'],
        'test_recall': test_metrics['recall_micro'],
        'test_precision': test_metrics['precision_micro'],
        'test_f1': test_metrics['f1_micro'],
        'top5_accuracy': topk_results[5]['accuracy'],
        'deployment_recommendation': quality_assessment['recommendation']
    }
}

output_path = CONFIG['output_file']
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)


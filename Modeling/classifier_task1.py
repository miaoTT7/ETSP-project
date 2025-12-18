import numpy as np
import json
import os
import time
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics import precision_recall_fscore_support, hamming_loss
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm
import joblib
import xgboost as xgb


start_time = time.time()


def enhance_features(X, scaler=None):
    # 1. 标准化
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    # 2. 统计特征
    X_mean = X.mean(axis=1, keepdims=True)
    X_std = X.std(axis=1, keepdims=True)
    X_max = X.max(axis=1, keepdims=True)
    X_min = X.min(axis=1, keepdims=True)
    
    # 3. 二阶特征 (只用前100维，避免维度爆炸)
    n_squared_features = min(100, X.shape[1])
    X_squared = X[:, :n_squared_features] ** 2
    
    # 4. 合并所有特征
    X_enhanced = np.hstack([
        X_scaled,
        X_mean,
        X_std,
        X_max,
        X_min,
        X_squared
    ])
    
    return X_enhanced, scaler

def calculate_adaptive_scale_pos_weight(n_pos, n_neg, total_samples):
    """
    自适应计算scale_pos_weight
    对极度不平衡的类别给予更大权重
    """
    if n_pos == 0:
        return 1.0
    
    base_weight = n_neg / n_pos
    
    # 根据样本数量调整权重
    if n_pos < 10:
        boost_factor = 2.0
    elif n_pos < 50:
        boost_factor = 1.5
    elif n_pos < 100:
        boost_factor = 1.2
    else:
        boost_factor = 1.0
    
    # 限制最大权重
    max_weight = 100.0
    return min(base_weight * boost_factor, max_weight)

def predict_xgb(models, X):
    """
    使用XGBoost模型预测
    返回每个标签的概率分数
    """
    n_samples = X.shape[0]
    n_labels = len(models)
    y_scores = np.zeros((n_samples, n_labels))
    
    dmatrix = xgb.DMatrix(X)
    
    for i, model in enumerate(models):
        if model is not None:
            y_scores[:, i] = model.predict(dmatrix)
    
    return y_scores

def calibrate_probabilities(y_val_scores, y_val, min_samples=10):
    """
    使用Isotonic Regression校准预测概率
    """
    calibrators = []
    
    print("📊 Calibrating probabilities...")
    
    for i in tqdm(range(y_val_scores.shape[1]), desc="Calibration"):
        y_true_label = y_val[:, i]
        y_score_label = y_val_scores[:, i]
        
        # 只对有足够正样本的标签进行校准
        if y_true_label.sum() >= min_samples:
            iso_reg = IsotonicRegression(out_of_bounds='clip')
            iso_reg.fit(y_score_label, y_true_label)
            calibrators.append(iso_reg)
        else:
            calibrators.append(None)
    
    return calibrators

def apply_calibration(y_scores, calibrators):
    """
    应用概率校准
    """
    y_scores_calibrated = y_scores.copy()
    
    for i, calibrator in enumerate(calibrators):
        if calibrator is not None:
            y_scores_calibrated[:, i] = calibrator.transform(y_scores[:, i])
    
    return y_scores_calibrated

def predict_with_adaptive_topk_threshold(y_scores, max_k=50, min_k=3, threshold=0.2):
    """
    自适应Top-K + 阈值组合策略
    - 在Top-K中选择概率>阈值的标签
    - 如果没有符合条件的，至少选Top-min_k
    """
    n_samples = y_scores.shape[0]
    y_pred = np.zeros_like(y_scores, dtype=int)
    
    for i in range(n_samples):
        # 获取Top-max_k的索引和分数
        top_k_indices = np.argsort(y_scores[i])[-max_k:]
        top_k_scores = y_scores[i, top_k_indices]
        
        # 从Top-K中筛选概率>阈值的
        valid_mask = top_k_scores >= threshold
        selected_indices = top_k_indices[valid_mask]
        
        # 如果筛选后太少，至少保留Top-min_k
        if len(selected_indices) < min_k:
            selected_indices = top_k_indices[-min_k:]
        
        y_pred[i, selected_indices] = 1
    
    return y_pred

def grid_search_optimal_params(y_val_scores, y_val, verbose=True):
    """
    网格搜索最优的max_k, min_k, threshold组合
    """
    best_f1 = 0
    best_params = {}
    all_results = []
    
    if verbose:
        print("\nGrid Search for Optimal Parameters:")
        print(f"{'max_k':<8} {'min_k':<8} {'threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Avg_Pred':<10}")
    
    # 搜索空间
    max_k_values = [20, 30, 40, 50]
    min_k_values = [1, 3, 5]
    threshold_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    
    for max_k in max_k_values:
        for min_k in min_k_values:
            for threshold in threshold_values:
                y_pred = predict_with_adaptive_topk_threshold(
                    y_val_scores, 
                    max_k=max_k, 
                    min_k=min_k, 
                    threshold=threshold
                )
                
                p, r, f1, _ = precision_recall_fscore_support(
                    y_val, y_pred, average='micro', zero_division=0
                )
                
                avg_predictions = y_pred.sum(axis=1).mean()
                
                all_results.append({
                    'max_k': max_k,
                    'min_k': min_k,
                    'threshold': threshold,
                    'precision': float(p),
                    'recall': float(r),
                    'f1': float(f1),
                    'avg_predictions': float(avg_predictions)
                })
                
                if verbose:
                    print(f"{max_k:<8} {min_k:<8} {threshold:<12.2f} {p:<12.4f} {r:<12.4f} {f1:<12.4f} {avg_predictions:<10.2f}")
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_params = {
                        'max_k': max_k,
                        'min_k': min_k,
                        'threshold': threshold,
                        'f1': f1,
                        'precision': p,
                        'recall': r,
                        'avg_predictions': avg_predictions
                    }
    
    if verbose:
        print("\nBest Parameters Found:")
        print(f"   max_k={best_params['max_k']}, min_k={best_params['min_k']}, threshold={best_params['threshold']:.2f}")
        print(f"   Precision: {best_params['precision']:.4f}")
        print(f"   Recall: {best_params['recall']:.4f}")
        print(f"   F1: {best_params['f1']:.4f}")
        print(f"   Avg Predictions: {best_params['avg_predictions']:.2f}")
    
    return best_params, all_results


# load data
train_embeddings_path = '../processed_data/TIBKAT/embedding/tibkat_train_embeddings.npy'
train_ids_path = '../processed_data/TIBKAT/embedding/tibkat_train_ids.json'
train_labels_path = '../processed_data/TIBKAT/translating/train_all.jsonl'

X_all = np.load(train_embeddings_path)
all_ids = json.load(open(train_ids_path, 'r'))

print(f"Total embeddings shape: {X_all.shape}")
print(f"Total papers: {len(all_ids):,}")

# load labels
all_labels = {}
with open(train_labels_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        paper_id = data['paper_id']
        subjects = data.get('subject', {}).get('labels', [])
        all_labels[paper_id] = subjects

# align labels with IDs
y_all_raw = []
for paper_id in all_ids:
    y_all_raw.append(all_labels.get(paper_id, []))

print(f"Loaded labels for {len(all_labels):,} papers")


# 2. Split train/validation sets

print("\nSplitting train/validation sets...")

X_train, X_val, y_train_raw, y_val_raw, train_ids, val_ids = train_test_split(
    X_all, y_all_raw, all_ids,
    test_size=3000,
    random_state=42,
    shuffle=True
)

print(f"✓ Training samples: {len(X_train):,}")
print(f"✓ Validation samples: {len(X_val):,}")

# 3. Feature enhancement

print("\nEnhancing features...")

X_train_enhanced, feature_scaler = enhance_features(X_train)
X_val_enhanced, _ = enhance_features(X_val, feature_scaler)

print(f"✓ Original feature dim: {X_train.shape[1]}")
print(f"✓ Enhanced feature dim: {X_train_enhanced.shape[1]}")

# 4. Binarize multi-labels

print("\nBinarizing multi-labels...")

mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(y_train_raw)
y_val = mlb.transform(y_val_raw)

n_labels = y_train.shape[1]

print(f"Number of unique subjects: {len(mlb.classes_):,}")
print(f"Training label matrix: {y_train.shape}")
print(f"Validation label matrix: {y_val.shape}")
print(f"Average labels per paper (train): {y_train.sum(axis=1).mean():.2f}")

# 保存模型相关文件
models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)

mlb_path = os.path.join(models_dir, 'label_binarizer.pkl')
joblib.dump(mlb, mlb_path)

scaler_path = os.path.join(models_dir, 'feature_scaler.pkl')
joblib.dump(feature_scaler, scaler_path)

print(f"Label binarizer saved to {mlb_path}")
print(f"Feature scaler saved to {scaler_path}")

# Analyze label distribution
label_counts = y_train.sum(axis=0)
print(f"\nLabel Distribution:")
print(f"   Most frequent: {label_counts.max():,} papers")
print(f"   Least frequent: {label_counts.min():,} papers")
print(f"   Median: {int(np.median(label_counts)):,} papers")
print(f"   Labels with <10 samples: {(label_counts < 10).sum():,}")


# Train XGBoost models

print("\n🤖 Training XGBoost models...")

xgb_models = []
training_stats = []

# Optimized XGBoost parameters
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    
    # Tree structure parameters
    'max_depth': 8,                    # 增加深度
    'min_child_weight': 1,             # 降低最小权重
    'gamma': 0.05,                     # 降低gamma
    
    # Learning parameters
    'learning_rate': 0.03,             # 降低学习率
    
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'colsample_bylevel': 0.8,
    
    'reg_alpha': 0.05,
    'reg_lambda': 0.05,
    
    'tree_method': 'hist',
    'n_jobs': -1,
    'random_state': 42
}

print(f"Training {n_labels} binary classifiers...")
print(f"Hyperparameters: {json.dumps(xgb_params, indent=2)}")

train_start = time.time()

for i in tqdm(range(n_labels), desc="Training"):
    y_train_label = y_train[:, i]
    
    n_pos = y_train_label.sum()
    n_neg = len(y_train_label) - n_pos
    
    # Skip labels with too few samples
    if n_pos < 5:
        xgb_models.append(None)
        training_stats.append({
            'label_index': i,
            'label_name': mlb.classes_[i],
            'n_positive': int(n_pos),
            'trained': False,
            'reason': 'too_few_samples'
        })
        continue
    
    scale_pos_weight = calculate_adaptive_scale_pos_weight(
        n_pos, n_neg, len(y_train_label)
    )
    
    current_params = xgb_params.copy()
    current_params['scale_pos_weight'] = scale_pos_weight
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X_train_enhanced, label=y_train_label)
    dval = xgb.DMatrix(X_val_enhanced, label=y_val[:, i])
    
    # Train
    evals = [(dtrain, 'train'), (dval, 'eval')]
    
    model = xgb.train(
        current_params,
        dtrain,
        num_boost_round=300,
        evals=evals,
        early_stopping_rounds=20,
        verbose_eval=False
    )
    
    xgb_models.append(model)
    
    training_stats.append({
        'label_index': i,
        'label_name': mlb.classes_[i],
        'n_positive': int(n_pos),
        'n_negative': int(n_neg),
        'scale_pos_weight': float(scale_pos_weight),
        'best_iteration': int(model.best_iteration),
        'best_score': float(model.best_score),
        'trained': True
    })

train_time = time.time() - train_start

# Save models
xgb_model_path = os.path.join(models_dir, 'xgboost_models_optimized.pkl')
joblib.dump(xgb_models, xgb_model_path)

stats_path = os.path.join(models_dir, 'training_stats_optimized.json')
with open(stats_path, 'w', encoding='utf-8') as f:
    json.dump(training_stats, f, indent=2, ensure_ascii=False)


n_trained = sum(1 for s in training_stats if s['trained'])

# 6. Validation set prediction


predict_start = time.time()
y_val_scores = predict_xgb(xgb_models, X_val_enhanced)
predict_time = time.time() - predict_start

# 7. Probability calibration
calibrators = calibrate_probabilities(y_val_scores, y_val, min_samples=10)
y_val_scores_calibrated = apply_calibration(y_val_scores, calibrators)

# Save calibrators
calibrators_path = os.path.join(models_dir, 'probability_calibrators.pkl')
joblib.dump(calibrators, calibrators_path)

n_calibrated = sum(1 for c in calibrators if c is not None)

# 8. Searching for optimal prediction parameters


best_params, search_results = grid_search_optimal_params(
    y_val_scores_calibrated, 
    y_val,
    verbose=True
)

# 9. Test set evaluation

test_embeddings_path = '../processed_data/TIBKAT/embedding/tibkat_test_embeddings.npy'
test_ids_path = '../processed_data/TIBKAT/embedding/tibkat_test_ids.json'
test_labels_path = '../processed_data/TIBKAT/translating/test_all.jsonl'

if os.path.exists(test_embeddings_path):
    X_test = np.load(test_embeddings_path)
    test_ids = json.load(open(test_ids_path, 'r'))
    
    # Load test labels
    test_labels = {}
    with open(test_labels_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            test_labels[data['paper_id']] = data.get('subject', {}).get('labels', [])
    
    y_test_raw = [test_labels.get(pid, []) for pid in test_ids]
    y_test = mlb.transform(y_test_raw)
    
    
    # Enhance features
    X_test_enhanced, _ = enhance_features(X_test, feature_scaler)
    
    # Predict
    y_test_scores = predict_xgb(xgb_models, X_test_enhanced)
    y_test_scores_calibrated = apply_calibration(y_test_scores, calibrators)
    
    # Predict using optimal parameters
    y_test_pred = predict_with_adaptive_topk_threshold(
        y_test_scores_calibrated,
        max_k=best_params['max_k'],
        min_k=best_params['min_k'],
        threshold=best_params['threshold']
    )
    
    # Evaluation
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average='micro', zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average='macro', zero_division=0
    )
    hamming_test = hamming_loss(y_test, y_test_pred)
    
    avg_predictions = y_test_pred.sum(axis=1).mean()

    results = {
        'model_info': {
            'model_type': 'XGBoost (Optimized with Feature Enhancement & Calibration)',
            'n_labels': n_labels,
            'n_train_samples': len(X_train),
            'n_val_samples': len(X_val),
            'n_test_samples': len(X_test),
            'original_feature_dim': X_train.shape[1],
            'enhanced_feature_dim': X_train_enhanced.shape[1],
            'training_time_seconds': train_time,
            'training_time_minutes': round(train_time / 60, 2),
            'prediction_time_seconds': predict_time
        },
        'hyperparameters': xgb_params,
        'optimization': {
            'best_params': best_params,
            'grid_search_results': search_results
        },
        'test_results': {
            'micro': {
                'precision': float(p_micro),
                'recall': float(r_micro),
                'f1': float(f1_micro)
            },
            'macro': {
                'precision': float(p_macro),
                'recall': float(r_macro),
                'f1': float(f1_macro)
            },
            'hamming_loss': float(hamming_test),
            'avg_predictions': float(avg_predictions)
        },
        'label_statistics': {
            'n_trained_models': n_trained,
            'n_skipped_models': n_labels - n_trained,
            'n_calibrated_models': n_calibrated,
            'avg_positive_samples': float(np.mean([s['n_positive'] for s in training_stats if s['trained']])),
            'median_positive_samples': float(np.median([s['n_positive'] for s in training_stats if s['trained']]))
        }
    }
    
    results_path = os.path.join(models_dir, 'optimized_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
else:
    print("\nNo test set found. Skipping test evaluation.")



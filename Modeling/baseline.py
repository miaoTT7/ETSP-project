import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
import time
from tqdm import tqdm
import random

start_time = time.time()

# GND subjects
subject_embeddings = np.load('../processed_data/GND/embedding/subject_embeddings.npy')
subject_ids = json.load(open('../processed_data/GND/embedding/subject_ids.json', 'r'))
subject_texts_raw = json.load(open('../processed_data/GND/embedding/subject_texts.json', 'r'))
print(f"GND subjects: {len(subject_ids):,}")

# Convert list to dict: {subject_id: text}
if isinstance(subject_texts_raw, dict):
    subject_texts = subject_texts_raw
else:
    subject_texts = {subject_ids[i]: subject_texts_raw[i] for i in range(len(subject_ids))}

print(f"GND subjects: {len(subject_ids):,}")

# TIBKAT test papers
test_embeddings = np.load('../processed_data/TIBKAT/embedding/tibkat_test_embeddings.npy')
test_ids = json.load(open('../processed_data/TIBKAT/embedding/tibkat_test_ids.json', 'r'))
print(f"Test papers: {len(test_ids):,}")

# True labels
true_labels = {}
paper_data = {}
with open('../processed_data/TIBKAT/translating/test_all.jsonl', 'r') as f:
    for line in f:
        paper = json.loads(line)
        paper_id = paper['paper_id']
        true_labels[paper_id] = paper['subject']['labels']
        paper_data[paper_id] = {
            'title': paper.get('title', ''),
            'abstract': paper.get('content', {}).get('text', '')
        }

# Data coverage
tibkat_subjects = set()
for labels in true_labels.values():
    tibkat_subjects.update(labels)
available_in_gnd = tibkat_subjects.intersection(set(subject_ids))
coverage = len(available_in_gnd) / len(tibkat_subjects) if tibkat_subjects else 0

def evaluate_subject_indexing(k_values=[1, 3, 5], batch_size=500):
    # Find valid papers
    valid_indices = []
    valid_true_labels = []
    
    for i, paper_id in enumerate(test_ids):
        if paper_id in true_labels:
            valid_indices.append(i)
            valid_true_labels.append(set(true_labels[paper_id]))
    
    total_papers = len(valid_indices)
    print(f"Evaluating {total_papers:,} papers")
    
    if total_papers == 0:
        return {}
    
    valid_embeddings = test_embeddings[valid_indices]
    max_k = max(k_values)
    
    results = {k: {'precisions': [], 'recalls': []} for k in k_values}
    predictions = {}  # Store for later use
    
    for batch_start in tqdm(range(0, total_papers, batch_size), desc="Processing"):
        batch_end = min(batch_start + batch_size, total_papers)
        batch_embeddings = valid_embeddings[batch_start:batch_end]
        batch_true_labels = valid_true_labels[batch_start:batch_end]
        
        batch_similarities = cosine_similarity(batch_embeddings, subject_embeddings)
        topk_indices = np.argpartition(batch_similarities, -max_k, axis=1)[:, -max_k:]
        
        for i in range(len(batch_similarities)):
            sorted_topk = topk_indices[i][np.argsort(batch_similarities[i, topk_indices[i]])[::-1]]
            true_subs = batch_true_labels[i]
            paper_id = test_ids[valid_indices[batch_start + i]]
            
            # Store predictions
            predictions[paper_id] = [subject_ids[idx] for idx in sorted_topk[:max(k_values)]]
            
            for k in k_values:
                predicted_set = set(subject_ids[idx] for idx in sorted_topk[:k])
                intersection = predicted_set & true_subs
                precision = len(intersection) / k
                recall = len(intersection) / len(true_subs) if true_subs else 0
                
                results[k]['precisions'].append(precision)
                results[k]['recalls'].append(recall)
    
    final_results = {}
    for k in k_values:
        p = np.mean(results[k]['precisions'])
        r = np.mean(results[k]['recalls'])
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        
        final_results[k] = {
            'precision': p,
            'recall': r,
            'f1': f1,
            'papers_evaluated': total_papers
        }
    
    return final_results, predictions

indexing_results, subject_predictions = evaluate_subject_indexing(k_values=[1, 3, 5])

print("\nResults:")
for k in [1, 3, 5]:
    print(f"\nTop-{k}:")
    print(f"  Precision@{k}: {indexing_results[k]['precision']:.4f}")
    print(f"  Recall@{k}: {indexing_results[k]['recall']:.4f}")
    print(f"  F1@{k}: {indexing_results[k]['f1']:.4f}")


# PURPOSE 2: Paper Recommendation
def evaluate_paper_recommendation(k_values=[5, 10, 20], batch_size=500):
    # Find valid papers
    valid_indices = []
    valid_true_labels = []
    
    for i, paper_id in enumerate(test_ids):
        if paper_id in true_labels:
            valid_indices.append(i)
            valid_true_labels.append(set(true_labels[paper_id]))
    
    total_papers = len(valid_indices)
    print(f"Evaluating {total_papers:,} papers...")
    
    if total_papers == 0:
        return {}
    
    valid_embeddings = test_embeddings[valid_indices]
    max_k = max(k_values)
    
    results = {k: {'accuracies': []} for k in k_values}
    
    for batch_start in tqdm(range(0, total_papers, batch_size), desc="Processing"):
        batch_end = min(batch_start + batch_size, total_papers)
        batch_embeddings = valid_embeddings[batch_start:batch_end]
        batch_true_labels = valid_true_labels[batch_start:batch_end]
        
        # Calculate similarity between papers (exclude self)
        batch_similarities = cosine_similarity(batch_embeddings, valid_embeddings)
        
        for i in range(len(batch_similarities)):
            query_idx = batch_start + i
            similarities = batch_similarities[i].copy()
            similarities[query_idx] = -np.inf  # Exclude self
            
            # Get top-k most similar papers
            topk_indices = np.argpartition(similarities, -max_k)[-max_k:]
            sorted_topk = topk_indices[np.argsort(similarities[topk_indices])[::-1]]
            
            query_subjects = batch_true_labels[i]
            
            for k in k_values:
                # Check how many of top-k papers share at least one subject
                shared_count = 0
                for rec_idx in sorted_topk[:k]:
                    rec_subjects = valid_true_labels[rec_idx]
                    if query_subjects & rec_subjects:  # Has overlap
                        shared_count += 1
                
                accuracy = shared_count / k
                results[k]['accuracies'].append(accuracy)
    
    final_results = {}
    for k in k_values:
        final_results[k] = {
            'accuracy': np.mean(results[k]['accuracies']),
            'papers_evaluated': total_papers
        }
    
    return final_results

recommendation_results = evaluate_paper_recommendation(k_values=[5, 10, 20])

for k in [5, 10, 20]:
    print(f"\nTop-{k}:")
    print(f"  Accuracy@{k}: {recommendation_results[k]['accuracy']:.4f}")

# PURPOSE 3: Explanation Generation
def generate_rule_based_explanation(paper_id, predicted_subjects, top_n=3):
    """Generate template-based explanation"""
    
    # Get subject names
    subject_names = []
    for subj in predicted_subjects[:top_n]:
        name = subject_texts.get(subj)
        if name:
            subject_names.append(name)
        else:
            subject_names.append(subj)  # Fallback to ID
    
    if not subject_names:
        return "No subjects could be assigned to this paper."
    
    
    # Template
    if len(subject_names) == 1:
        explanation = f"This paper is classified under the subject: {subject_names[0]}."
    elif len(subject_names) == 2:
        explanation = f"This paper is classified under the subjects: {subject_names[0]} and {subject_names[1]}."
    else:
        subjects_str = ", ".join(subject_names[:-1]) + f", and {subject_names[-1]}"
        explanation = f"This paper is classified under the subjects: {subjects_str}."
    
    explanation += f" These subjects were selected based on semantic similarity to the paper's content."
    
    return explanation

# Generate sample explanations
print("\nGenerating sample explanations...")
sample_papers = random.sample([pid for pid in subject_predictions.keys()], min(5, len(subject_predictions)))

explanations = {}
for paper_id in sample_papers:
    predicted = subject_predictions[paper_id][:5]
    explanation = generate_rule_based_explanation(paper_id, predicted, top_n=3)
    explanations[paper_id] = {
        'predicted_subjects': predicted,
        'explanation': explanation
    }
    
    print(f"\n--- Paper: {paper_id} ---")
    print(f"Title: {paper_data[paper_id]['title'][:100]}...")
    print(f"Predicted: {[subject_texts.get(s, s) for s in predicted[:3]]}")
    print(f"Explanation: {explanation}")

# Save Results
end_time = time.time()
total_minutes = (end_time - start_time) / 60

output = {
    'method': 'baseline',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'evaluation_time_minutes': total_minutes,
    
    'purpose_1_subject_indexing': {
        'method': 'embedding_based_retrieval',
        'results': indexing_results
    },
    
    'purpose_2_paper_recommendation': {
        'method': 'embedding_similarity',
        'results': recommendation_results
    },
    
    'purpose_3_explanation_generation': {
        'method': 'rule_based_templates',
        'sample_explanations': explanations
    },
    
    'data_coverage': {
        'tibkat_subjects': len(tibkat_subjects),
        'available_in_gnd': len(available_in_gnd),
        'coverage': coverage
    }
}

with open('complete_baseline_results.json', 'w') as f:
    json.dump(output, f, indent=2)
from urllib.parse import unquote
import numpy as np
import json
import joblib
import random
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import torch
import xgboost as xgb
load_dotenv()

RAG_CONFIG={
    'backend':'huggingface',
    'huggingface':{
        'model_name':'google/flan-t5-large',
        'device':'mps',
        'max_length':200,
        'temperature':0.7,
        'token':os.environ.get('HUGGING_FACE_HUB_TOKEN',None)
    }
}

class PaperRecommendationEngine:
    def __init__(self,
                 models_dir:str='models',
                 data_dir:str='../processed_data/TIBKAT',
                 gnd_dir:str='../processed_data/GND'
                 ):
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.gnd_dir = gnd_dir
        

        self._load_models()
        self._load_train_data()
        self._load_subject_texts()
        self._precompute_train_predictions()

    def _load_models(self):
        """加载预训练的模型"""
        # MultiLabelBinarizer (先加载以获取类别数)
        self.mlb = joblib.load(f'{self.models_dir}/label_binarizer.pkl')
        print(f"✓ Loaded MultiLabelBinarizer with {len(self.mlb.classes_)} classes")
        
        # XGBoost 模型
        self.xgb_models = joblib.load(f'{self.models_dir}/xgboost_models_optimized.pkl')
        
        
        # 特征缩放器
        self.feature_scaler = joblib.load(f'{self.models_dir}/feature_scaler.pkl')
        print(f"✓ Loaded feature scaler")
        
        # 概率校准器
        self.calibrators = joblib.load(f'{self.models_dir}/probability_calibrators.pkl')
        print(f"✓ Loaded calibrators")

    def _load_gnd_mapping(self) -> Dict[str, str]:
        """加载 GND ID → 名称的映射"""
        gnd_mapping = {}
        path = f'{self.gnd_dir}/translating/translated_6_GND.jsonl'
        
        if not os.path.exists(path):
            print(f"⚠️ Warning: GND file not found at {path}")
            return gnd_mapping
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    gnd_id = record.get('Code', '')
                    name = record.get('Name', '')
                    if gnd_id and name:
                        gnd_mapping[gnd_id] = name
                except json.JSONDecodeError:
                    continue
        
        print(f"✓ Loaded {len(gnd_mapping):,} GND mappings")
        return gnd_mapping
    
    def _load_train_data(self):
        from urllib.parse import unquote
        emb_path = f'{self.data_dir}/embedding/tibkat_train_embeddings.npy'
        self.X_train = np.load(emb_path)
        print(f"✓ Loaded embeddings: {self.X_train.shape}")
        
        ids_path = f'{self.data_dir}/embedding/tibkat_train_ids.json'
        with open(ids_path, 'r', encoding='utf-8') as f:
            train_ids_raw = json.load(f)       
        self.train_ids = [unquote(str(pid)) for pid in train_ids_raw]
        # 加载 metadata
        self.train_metadata = {}
        metadata_path = f'{self.data_dir}/translating/train_all.jsonl'
        
        self.train_ids=[unquote(str(pid)) for pid in self.train_ids]
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                paper_id = unquote(str(data.get('paper_id', '')))
                
                content=data.get('content',{})
                metadata_info=data.get('metadata',{})
                subject_info=data.get('subject',{})

                self.train_metadata[paper_id] = {
                    'title': content.get('title_en', 'Unknown'),
                    'text': content.get('text', ''),
                    'authors': metadata_info.get('authors', 'Unknown'),
                    'year': metadata_info.get('date', 'Unknown'),
                    'type': metadata_info.get('type', 'Unknown'),
                    'subjects': subject_info.get('subject', {}).get('labels', []),
                    'subject_names': subject_info.get('names', [])
                }
        
        print(f"✓ Loaded {len(self.train_metadata):,} metadata entries")    

    def _load_subject_texts(self):
        self.subject_texts = {}
        path = f'{self.gnd_dir}/translating/translated_6_GND.jsonl'
        
        if not os.path.exists(path):
            print(f"GND subject texts not found")
            return
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    gnd_id = record.get('Code', '')
                    name = record.get('Name', '')
                    definition = record.get('Definition', '')
                    
                    if gnd_id:
                        text = f"{name}. {definition}".strip()
                        self.subject_texts[gnd_id] = text if text else name
                except json.JSONDecodeError:
                    continue
        
        print(f"✓ Loaded {len(self.subject_texts):,} subject descriptions")
    def _enhance_features(self, X: np.ndarray) -> np.ndarray:
        # 1. 标准化
        X_scaled = self.feature_scaler.transform(X)
        
        # 2. 统计特征
        X_mean = X.mean(axis=1, keepdims=True)
        X_std = X.std(axis=1, keepdims=True)
        X_max = X.max(axis=1, keepdims=True)
        X_min = X.min(axis=1, keepdims=True)
        
        # 3. 平方特征
        n_squared_features = min(100, X.shape[1])
        X_squared = X[:, :n_squared_features] ** 2
        
        # 4. 合并
        X_enhanced = np.hstack([X_scaled, X_mean, X_std, X_max, X_min, X_squared])
        
        return X_enhanced
    def _precompute_train_predictions(self):
        print("Precomputing train predictions...")
        
        X_train_enhanced = self._enhance_features(self.X_train)
        dmatrix = xgb.DMatrix(X_train_enhanced)
        
        self.y_train_pred_scores = np.zeros((len(self.X_train), len(self.xgb_models)))
        
        for i, model in enumerate(self.xgb_models):
            if model is not None:
                self.y_train_pred_scores[:, i] = model.predict(dmatrix)
            
            if (i + 1) % 2000 == 0:
                print(f"  Predicted {i+1}/{len(self.xgb_models)} labels")
        
        print("Applying calibration...")
        for i, calibrator in enumerate(self.calibrators):
            if calibrator is not None:
                self.y_train_pred_scores[:, i] = calibrator.transform(
                    self.y_train_pred_scores[:, i]
                )
        
        print("✓ Train predictions precomputed")
    
    def _predict_labels(self, X: np.ndarray) -> np.ndarray:

        X_enhanced = self._enhance_features(X)
        
        dmatrix = xgb.DMatrix(X_enhanced)
        
        y_scores = np.zeros((len(X), len(self.xgb_models)))
        
        for i, model in enumerate(self.xgb_models):
            if model is not None:
                y_scores[:, i] = model.predict(dmatrix)
        
        for i, calibrator in enumerate(self.calibrators):
            if calibrator is not None:
                y_scores[:, i] = calibrator.transform(y_scores[:, i])
        
        return y_scores
    
    def _calculate_label_similarity(
        self, 
        labels1: np.ndarray, 
        labels2: np.ndarray
    ) -> float:
        norm1 = np.linalg.norm(labels1)
        norm2 = np.linalg.norm(labels2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(labels1, labels2) / (norm1 * norm2))
    
    def _hybrid_similarity(
        self,
        query_embedding: np.ndarray,
        query_labels: np.ndarray,
        candidate_embeddings: np.ndarray,
        candidate_labels: np.ndarray,
        alpha: float = 0.6
    ) -> np.ndarray:

        emb_similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            candidate_embeddings
        )[0]
        
        label_similarities = np.array([
            self._calculate_label_similarity(query_labels, cand_labels)
            for cand_labels in candidate_labels
        ])
        
        hybrid_sim = alpha * label_similarities + (1 - alpha) * emb_similarities
        
        return hybrid_sim
    
    def recommend(
        self,
        query_embedding: np.ndarray,
        K: int = 10,
        label_threshold: float = 0.01,
        min_common_labels: int = 1,
        max_candidates: int = 1000,
        alpha: float = 0.6,
        return_metadata: bool = True
    ) -> Tuple[List[str], List[str], List[Dict]]:

        query_pred_scores = self._predict_labels(
            query_embedding.reshape(1, -1)
        )[0]
        
        top_10_indices = np.argsort(query_pred_scores)[-10:][::-1]
        predicted_subject_ids = self.mlb.classes_[top_10_indices].tolist()
        
        gnd_mapping = self._load_gnd_mapping()
        predicted_subject_names = [
            gnd_mapping.get(sid, sid) for sid in predicted_subject_ids
        ]
        
        query_labels_binary = (query_pred_scores >= label_threshold).astype(int)
        train_labels_binary = (self.y_train_pred_scores >= label_threshold).astype(int)
        
        common_label_counts = np.sum(
            query_labels_binary * train_labels_binary, axis=1
        )
        
        candidate_mask = common_label_counts >= min_common_labels
        candidate_indices = np.where(candidate_mask)[0]
        
        print(f"Candidate pool: {len(candidate_indices):,} papers with ≥{min_common_labels} common labels")
        
        if len(candidate_indices) > max_candidates:
            top_common_indices = np.argsort(common_label_counts)[-max_candidates:]
            candidate_indices = top_common_indices
            print(f"   → Reduced to {max_candidates:,} candidates")
        
        if len(candidate_indices) < K:
            print(f"Only {len(candidate_indices)} candidates (< K={K}), using full corpus")
            candidate_indices = np.arange(len(self.X_train))
        
        candidate_embeddings = self.X_train[candidate_indices]
        candidate_labels = self.y_train_pred_scores[candidate_indices]
        
        similarities = self._hybrid_similarity(
            query_embedding,
            query_pred_scores,
            candidate_embeddings,
            candidate_labels,
            alpha=alpha
        )
        
        if len(similarities) >= K:
            top_k_in_candidates = np.argsort(similarities)[-K:][::-1]
        else:
            top_k_in_candidates = np.argsort(similarities)[::-1]
        
        recommended_indices = candidate_indices[top_k_in_candidates]
        recommended_similarities = similarities[top_k_in_candidates]
        
        recommendations = []
        
        for rank, (idx, sim) in enumerate(zip(recommended_indices, recommended_similarities), 1):
            paper_id = self.train_ids[idx]
            
            cosine_sim = cosine_similarity(
                query_embedding.reshape(1, -1),
                self.X_train[idx].reshape(1, -1)
            )[0][0]
            
            label_sim = self._calculate_label_similarity(
                query_pred_scores,
                self.y_train_pred_scores[idx]
            )
            
            candidate_labels_binary = (self.y_train_pred_scores[idx] >= label_threshold).astype(int)
            common_labels_indices = np.where(query_labels_binary & candidate_labels_binary)[0]
            common_labels = [self.mlb.classes_[i] for i in common_labels_indices]
            
            rec = {
                'rank': rank,
                'paper_id': paper_id,
                'similarity': float(sim),
                'cosine_similarity': float(cosine_sim),
                'label_similarity': float(label_sim),
                'common_labels': common_labels,
                'common_labels_count': len(common_labels)
            }
            
            if return_metadata and paper_id in self.train_metadata:
                metadata = self.train_metadata[paper_id]
                rec.update({
                    'title': metadata.get('title', 'Unknown'),
                    'authors': metadata.get('authors', 'Unknown'),
                    'year': metadata.get('year', 'Unknown'),
                    'subjects': metadata.get('subjects', []),
                    'subject_names': metadata.get('subject_names', [])
                })
            
            recommendations.append(rec)
        
        return predicted_subject_ids, predicted_subject_names, recommendations
    
    def get_paper_metadata(self, paper_id: str) -> Optional[Dict]:
        return self.train_metadata.get(paper_id, None)


class RAGExplainerBackend:
    def generate_explanation(self, query_info: str, recommendation: str,context: str) -> str:
        raise NotImplementedError

class HuggingFaceBackend(RAGExplainerBackend):
    
    def __init__(self, config:Dict):
        self.config=config
        self.model=None
        self.tokenizer=None

        self._load_model()

    
    def _load_model(self):
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            import torch
            model_name = self.config.get('model_name', 'google/flan-t5-base')
            device = self.config.get('device', 'cpu')
            token = self.config.get('token', None)
            
            self.tokenizer=AutoTokenizer.from_pretrained(model_name, token=token)
            self.model=AutoModelForSeq2SeqLM.from_pretrained(model_name, token=token).to(device)

            # device
            if device =='mps':
                if torch.backends.mps.is_available():
                    self.model=self.model.to(device='mps')
                else:
                    print("MPS device not available, falling back to CPU")
                    self.model=self.model.to(device='cpu')
            elif device == 'cuda':  
                if torch.cuda.is_available():
                    self.model=self.model.to(device='cuda')
                elif torch.backends.mps.is_available():
                    self.model=self.model.to(device='mps')
                else:
                    self.model=self.model.to(device='cpu')
            else:
                self.model=self.model.to(device='cpu')
        except ImportError:
            print("transformers not installed. Install with: pip install transformers")
            raise
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    def _load_model(self):
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            import torch
            
            model_name = self.config.get('model_name', 'google/flan-t5-base')
            device = self.config.get('device', 'cpu')
            token = self.config.get('token', None)
            
            print(f"Loading {model_name} on {device}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=token)
            
            if device == 'mps':
                if torch.backends.mps.is_available():
                    self.model = self.model.to('mps')
                    print("✓ Using MPS (Apple Silicon)")
                else:
                    print("MPS not available, falling back to CPU")
                    self.model = self.model.to('cpu')
            
            elif device == 'cuda':
                if torch.cuda.is_available():
                    self.model = self.model.to('cuda')
                    print("✓ Using CUDA (GPU)")
                else:
                    print("CUDA not available, falling back to CPU")
                    self.model = self.model.to('cpu')
            
            else:  # device == 'cpu' or any other value
                self.model = self.model.to('cpu')
                print("Using CPU")
            
            print("Model loaded successfully")
            
        except ImportError:
            print("transformers not installed. Install with: pip install transformers")
            raise
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    def generate_explanation(self, query_info: Dict, recommendation: Dict, context: str) -> str:
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model or tokenizer not loaded properly.")
        
        try:
            # construct prompt
            prompt = self._construct_prompt(query_info, recommendation,context )
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            )
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            max_length = self.config.get('max_length', 200)
            temperature = self.config.get('temperature', 0.7)

            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                num_return_sequences=1
            )
            
            # Decode
            explanation = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )  
            return explanation.strip()
        except Exception as e:
            print(f"Error generating explanation: {e}")
            return "Explanation could not be generated."          
        

    def _construct_prompt(self, query_info: Dict, recommendation: Dict, context: str) -> str:
        query_title = query_info.get('title', 'Unknown')
        rec_title = recommendation.get('title', 'Unknown')
        rec_authors = recommendation.get('authors', 'Unknown')
        rec_year = recommendation.get('year', 'Unknown')
        similarity = recommendation.get('similarity', 0)
        common_count = recommendation.get('common_labels_count', 0)   

        prompt=f"""Explain why this paper is recommended:

Query Paper: {query_title}

Recommended Paper: {rec_title}
Authors: {rec_authors}
Year: {rec_year}
Similarity: {similarity:.1%}
Common Topics: {common_count}

Context: {context}

Provide a brief explanation (2-3 sentences) of why this paper is relevant to the query paper."""
        
        return prompt
       

    def _fallback_explanation(
        self,
        query_info: Dict,
        recommendation: Dict,
        context: str
    ) -> str:
        """Fallback explanation when model fails"""
        similarity = recommendation.get('similarity', 0)
        common_count = recommendation.get('common_labels_count', 0)
        
        return (
            f"This paper is recommended with {similarity:.1%} similarity "
            f"based on {common_count} shared research topics. "
            f"The recommendation is based on semantic and topic-level analysis."
        )
    
class RAGExplainer:

    def __init__(self,gnd_dir:str='../processed_data/GND',llm_config:Optional[Dict]=None):
        self.gnd_dir=gnd_dir
        self.gnd_mapping=self._load_gnd_mappings()

        backend_type=RAG_CONFIG.get('backend','huggingface')
        if backend_type=='huggingface':
            self.backend=HuggingFaceBackend(RAG_CONFIG['huggingface'])
        else:
            raise ValueError(f"Unsupported backend type: {backend_type}")
    
    def _load_gnd_mappings(self) -> Dict[str, str]:
        gnd_mapping={}
        path = f'{self.gnd_dir}/translating/translated_6_GND.jsonl'
        with open(path,'r',encoding='utf-8') as f:
            for line in f:
                line=line
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    gnd_id = record.get('Code', '')
                    name = record.get('Name', '')
                    if gnd_id and name:
                        gnd_mapping[gnd_id] = name
                except json.JSONDecodeError:
                    continue
        return gnd_mapping
    def explain_recommendation(
        self,
        query_info: Dict,
        recommendation: Dict
    ) -> str:
        # Build context from common topics
        context = self._build_context(recommendation)
        
    
        explanation = self.backend.generate_explanation(
                query_info,
                recommendation,
                context
            )
        
        return explanation
    def _build_context(self, recommendation: Dict) -> str:
        common_labels = recommendation.get('common_labels', [])
        
        if not common_labels:
            return "No specific shared topics identified."
        
        # Get topic names
        topic_names = [
            self.gnd_mapping.get(label, label)
            for label in common_labels[:5]  # Top 5
        ]
        
        context = f"Shared research topics: {', '.join(topic_names)}"
        return context

    def _template_explanation(
        self,
        query_info: Dict,
        recommendation: Dict,
        context: str
    ) -> str:
        """Template-based explanation (fallback)"""
        rec_title = recommendation.get('title', 'this paper')
        rec_authors = recommendation.get('authors', 'Unknown')
        rec_year = recommendation.get('year', 'Unknown')
        similarity = recommendation.get('similarity', 0)
        cosine_sim = recommendation.get('cosine_similarity', 0)
        label_sim = recommendation.get('label_similarity', 0)
        common_count = recommendation.get('common_labels_count', 0)
        
        # Build explanation
        parts = []
        
        # Opening
        parts.append(
            f"The paper '{rec_title}' by {rec_authors} ({rec_year}) "
            f"is recommended with {similarity:.1%} overall similarity."
        )
        
        # Similarity breakdown
        if common_count > 0:
            parts.append(
                f"It shares {common_count} research topics with your query, "
                f"indicating strong thematic alignment (topic similarity: {label_sim:.1%})."
            )
        # Semantic similarity
        if cosine_sim > 0.5:
            parts.append(
                f"The semantic similarity ({cosine_sim:.1%}) suggests "
                f"overlap in research methodologies and concepts."
            )
        
        # Context
        if context and "No specific" not in context:
            parts.append(context + ".")
        
        return " ".join(parts)
    
class RecommendationSystemWithRAG:
    def __init__(
        self,
        models_dir: str = 'models',
        data_dir: str = '../processed_data/TIBKAT',
        gnd_dir: str = '../processed_data/GND',
        llm_config: Optional[Dict] = None
    ):
        self.engine = PaperRecommendationEngine(
            models_dir=models_dir,
            data_dir=data_dir,
            gnd_dir=gnd_dir
        )
        
        self.explainer = RAGExplainer(
            gnd_dir=gnd_dir,
            llm_config=llm_config
        )
    def recommend_and_explain(
        self,
        query_embedding: np.ndarray,
        query_info: Dict,
        K: int = 10,
        alpha: float = 0.6,
        label_threshold: float = 0.01
    ) -> Dict:
        
        predicted_subjects, predicted_subject_names, recommendations = self.engine.recommend(
            query_embedding=query_embedding,
            K=K,
            label_threshold=label_threshold,
            alpha=alpha,
            return_metadata=True
        )
        
        for rec in recommendations:
            rec['explanation'] = self.explainer.explain_recommendation(
                query_info, rec
            )
        
        return {
            'predicted_subjects': predicted_subjects,
            'predicted_subject_names': predicted_subject_names,
            'recommendations': recommendations
        }


if __name__ == "__main__": 
    dev_embeddings = np.load('../processed_data/TIBKAT/embedding/tibkat_test_embeddings.npy')

    with open('../processed_data/TIBKAT/embedding/tibkat_test_ids.json', 'r') as f:
        dev_ids = [unquote(str(pid)) for pid in json.load(f)]
    dev_metadata = {}
    with open('../processed_data/TIBKAT/translating/test_all.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            paper_id = unquote(str(data.get('paper_id', '')))
            content = data.get('content', {})
            
            dev_metadata[paper_id] = {
                'title': content.get('title_en', 'Unknown'),
                'text': content.get('text', '')
            }
    random.seed(42)
    N_SAMPLES = 20
    sampled_indices = random.sample(range(len(dev_ids)), N_SAMPLES)
    
    print("Loading recommendation system...")
    system = RecommendationSystemWithRAG(
        models_dir='models',
        data_dir='../processed_data/TIBKAT',
        gnd_dir='../processed_data/GND'    )
    results = []
    
    for i, idx in enumerate(sampled_indices, 1):
        paper_id = dev_ids[idx]
        metadata = dev_metadata.get(paper_id, {})
        
        print(f"[{i}/{N_SAMPLES}] Processing: {metadata.get('title', 'Unknown')[:60]}...")
        
        query_info = {
            'paper_id': paper_id,
            'title': metadata.get('title', 'Unknown'),
            'abstract': metadata.get('text', '')[:500]
        }
        
        K=5
        
        result = system.recommend_and_explain(
            query_embedding=dev_embeddings[idx],
            query_info=query_info,
            K=K
        )
        recommendations = []
        for rec in result['recommendations']:
            recommendations.append({
                'rank': rec['rank'],
                'paper_id': rec['paper_id'],
                'title': rec['title'],
                'authors': rec['authors'],
                'year': rec['year'],
                'similarity': float(rec['similarity']),
                'explanation': rec['explanation']
            })
        
        results.append({
            'query_paper_id': paper_id,
            'query_title': metadata.get('title', 'Unknown'),
            'recommendations': recommendations
        })
    output_path = 'models/task3_evaluation_results.json'
    Path(output_path).parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({'results': results}, f, indent=2, ensure_ascii=False)
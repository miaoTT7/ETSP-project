# import sys
# import os

# # ✅ 添加Modeling目录到sys.path，以便导入classifier_task3_RAG_explainer
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../Modeling'))

# from flask import Flask, render_template, request, jsonify
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from classifier_task3_RAG_explainer import *

# app = Flask(__name__)

# # ✅ 全局变量：加载模型（启动时加载一次）
# print("Initializing Paper Recommendation System...")

# # 1. 加载推荐系统（相对于web_app/app.py的路径）
# recommendation_system = RecommendationSystemWithRAG(
#     models_dir='../Modeling/models',
#     data_dir='../processed_data/TIBKAT',
#     gnd_dir='../processed_data/GND',
#     use_llm=True  # 使用LLM生成解释
# )


# embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# print("All models loaded! Server ready.")


# @app.route('/')
# def index():
#     """主页"""
#     return render_template('index.html')


# @app.route('/recommend', methods=['POST'])
# def recommend():
#     """推荐接口"""
#     try:
#         # 1. 获取用户输入
#         data = request.get_json()
#         title = data.get('title', '').strip()
#         abstract = data.get('abstract', '').strip()
#         K = int(data.get('K', 5))
#         alpha = float(data.get('alpha', 0.6))
        
#         # 验证输入
#         if not title:
#             return jsonify({'success': False, 'error': 'Title is required'}), 400
        
#         print(f"New Query:")
#         print(f"   Title: {title[:60]}...")
#         print(f"   Abstract: {abstract[:100] if abstract else '(None)'}...")
#         print(f"   K={K}, alpha={alpha}")
        
#         # 2. 生成Query Embedding
#         query_text = f"{title}. {abstract}" if abstract else title
        
#         print("Generating embedding...")
#         query_embedding = embedder.encode(query_text, convert_to_numpy=True)
#         print(f"Embedding shape: {query_embedding.shape}")
        
#         print("Predicting labels for the query...")
#         query_predicted_labels = recommendation_system.predict_labels(query_embedding)
#         label_threshold = 0.5
#         query_pred_labels_binary = (query_predicted_labels >= label_threshold).astype(int)
#         query_predicted_subject_ids=recommendation_system.engine.mlb.classes_[query_pred_labels_binary==1]
#         query_predicted_subjects = []
#         for subject_id in query_predicted_subject_ids:
#             subject_name = recommendation_system.explainer.subject_texts.get(subject_id, subject_id)
#             # 获取预测概率
#             subject_idx = np.where(recommendation_system.mlb.classes_ == subject_id)[0]
#             if len(subject_idx) > 0:
#                 probability = float(query_predicted_labels[subject_idx[0]])
#             else:
#                 probability = 0.0
            
#             query_predicted_subjects.append({
#                 'id': subject_id,
#                 'name': subject_name,
#                 'probability': probability
#             })
        
#         # 按概率降序排序
#         query_predicted_subjects = sorted(
#             query_predicted_subjects, 
#             key=lambda x: x['probability'], 
#             reverse=True
#         )

#         query_info = {
#             'paper_id': 'user_input',
#             'title': title,
#             'abstract': abstract,
#             'subjects': [s['id'] for s in query_predicted_subjects]
#         }
        
#         # 4. 调用推荐系统
#         print("Generating recommendations...")
#         result = recommendation_system.recommend_and_explain(
#             query_embedding=query_embedding,
#             query_info=query_info,
#             K=K,
#             alpha=alpha
#         )
        
#         print(f"✅ Generated {len(result['recommendations'])} recommendations")
        
#         # 5. 格式化返回结果
#         recommendations = []
#         for rec in result['recommendations']:
#             # 获取共同主题的可读名称
#             # common_topics_readable = []
#             # for label_id in rec['common_labels'][:5]:
#             #     topic_name = recommendation_system.explainer.subject_texts.get(
#             #         label_id, label_id
#             #     )
#             #     if len(topic_name) > 60:
#             #         topic_name = topic_name[:57] + '...'
#             #     common_topics_readable.append(topic_name)
#             rec_subjects = []
#             subject_ids = rec.get('subjects', [])
#             subject_names = rec.get('subject_names', [])
            
#             for idx, subject_id in enumerate(subject_ids):
#                 if idx < len(subject_names):
#                     name = subject_names[idx]
#                 else:
#                     # 如果 subject_names 不够长，从 GND 映射获取
#                     name = recommendation_system.gnd_mapping.get(subject_id, subject_id)
                
#                 rec_subjects.append({
#                     'id': subject_id,
#                     'name': name
#                 })
            
#             # ✅ 推荐论文的预测主题（从 XGBoost）
#             rec_predicted_subjects = []
#             for subject_id in rec.get('predicted_subjects', []):
#                 subject_name = recommendation_system.explainer.subject_texts.get(subject_id, subject_id)
#                 rec_predicted_subjects.append({
#                     'id': subject_id,
#                     'name': subject_name
#                 })
            
#             # ✅ 共同主题（带 ID 和名称）
#             common_topics = []
#             for label_id in rec['common_labels'][:5]:
#                 topic_name = recommendation_system.explainer.subject_texts.get(
#                     label_id, label_id
#                 )
#                 common_topics.append({
#                     'id': label_id,
#                     'name': topic_name[:60] if len(topic_name) > 60 else topic_name
#                 })
            
#             recommendations.append({
#                 'rank': rec['rank'],
#                 'title': rec['title'],
#                 'authors': rec['authors'],
#                 'year': rec['year'],
#                 'type': rec['type'],
#                 'similarity': f"{rec['similarity']:.2%}",
#                 'cosine_similarity': f"{rec['cosine_similarity']:.2%}",
#                 'label_similarity': f"{rec['label_similarity']:.2%}",
#                 'common_topics_count': len(rec['common_labels']),
#                 'common_topics': common_topics,
#                 'predicted_subjects': rec_predicted_subjects[:10],
#                 'subjects': rec_subjects,
#                 'similarity_type': rec['similarity_type'],
#                 'explanation': rec['explanation'],
#                 'text_preview': rec.get('text_preview', ''),
#             })
        
#         return jsonify({
#             'success': True,
#             'query': {
#                 'title': title,
#                 'abstract': abstract,
#                 'predicted_subjects': query_predicted_subjects[:10]
#             },
#             'recommendations': recommendations,
#             'config': result['config']
#         })
    
#     except Exception as e:
#         import traceback
#         error_trace = traceback.format_exc()
#         print(f"\n❌ Error occurred:")
#         print(error_trace)
        
#         return jsonify({
#             'success': False,
#             'error': str(e),
#             'trace': error_trace
#         }), 500


# @app.route('/health')
# def health():
#     """健康检查"""
#     return jsonify({
#         'status': 'healthy',
#         'model_loaded': recommendation_system is not None,
#         'embedder_loaded': embedder is not None
#     })


# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=8080)
import sys
import os

# 添加Modeling目录到sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../Modeling'))

from flask import Flask, render_template, request, jsonify
import numpy as np
from sentence_transformers import SentenceTransformer
from classifier_task3_RAG_explainer import RecommendationSystemWithRAG

app = Flask(__name__)

# 全局变量：加载模型
print("🚀 Initializing Paper Recommendation System with RAG...")

try:
    # 加载推荐系统
    recommendation_system = RecommendationSystemWithRAG(
        models_dir='../Modeling/models',
        data_dir='../processed_data/TIBKAT',
        gnd_dir='../processed_data/GND'
 )
    
    # 加载 sentence embedder
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    print("✅ All models loaded successfully!")
    print(f"   - Train data: {len(recommendation_system.engine.train_ids):,} papers")
    print(f"   - Metadata: {len(recommendation_system.engine.train_metadata):,} entries")
    print(f"   - GND mappings: {len(recommendation_system.explainer.gnd_mapping):,} subjects")
    
except Exception as e:
    print(f"❌ Error loading models: {e}")
    import traceback
    traceback.print_exc()
    recommendation_system = None
    embedder = None


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    """推荐接口"""
    try:
        # 检查模型是否加载
        if recommendation_system is None or embedder is None:
            return jsonify({
                'success': False,
                'error': 'Models not loaded. Please restart the server.'
            }), 500
        
        # 1. 获取用户输入
        data = request.get_json()
        title = data.get('title', '').strip()
        abstract = data.get('abstract', '').strip()
        K = int(data.get('K', 5))
        alpha = float(data.get('alpha', 0.6))
        
        # 验证输入
        if not title:
            return jsonify({'success': False, 'error': 'Title is required'}), 400
        
        print(f"\n{'='*60}")
        print(f"📝 New Query:")
        print(f"   Title: {title[:80]}...")
        if abstract:
            print(f"   Abstract: {abstract[:100]}...")
        print(f"   K={K}, alpha={alpha}")
        
        # 2. 生成 Query Embedding
        query_text = f"{title}. {abstract}" if abstract else title
        query_embedding = embedder.encode(query_text, convert_to_numpy=True)
        print(f"✓ Generated embedding: shape {query_embedding.shape}")
        
        # 3. 预测查询论文的主题
        print("🤖 Predicting subjects for query...")
        query_predicted_labels = recommendation_system.engine._predict_labels(
            query_embedding.reshape(1, -1)
        )[0]
        
        label_threshold = 0.5
        query_pred_labels_binary = (query_predicted_labels >= label_threshold).astype(int)
        query_predicted_subject_ids = recommendation_system.engine.mlb.classes_[
            query_pred_labels_binary == 1
        ]
        
        # 构建预测主题列表
        query_predicted_subjects = []
        for subject_id in query_predicted_subject_ids:
            subject_name = recommendation_system.explainer.gnd_mapping.get(
                subject_id, subject_id
            )
            subject_idx = np.where(recommendation_system.engine.mlb.classes_ == subject_id)[0]
            
            if len(subject_idx) > 0:
                probability = float(query_predicted_labels[subject_idx[0]])
            else:
                probability = 0.0
            
            query_predicted_subjects.append({
                'id': subject_id,
                'name': subject_name,
                'probability': probability
            })
        
        # 按概率降序排序
        query_predicted_subjects = sorted(
            query_predicted_subjects,
            key=lambda x: x['probability'],
            reverse=True
        )
        
        print(f"✓ Predicted {len(query_predicted_subjects)} subjects")
        
        # 4. 构建 query_info
        query_info = {
            'paper_id': 'user_query',
            'title': title,
            'abstract': abstract,
            'subjects': [s['id'] for s in query_predicted_subjects]
        }
        
        # 5. 调用推荐系统
        print("🔍 Generating recommendations with RAG explanations...")
        result = recommendation_system.recommend_and_explain(
            query_embedding=query_embedding,
            query_info=query_info,
            K=K,
            alpha=alpha
        )
        
        print(f"✅ Generated {len(result['recommendations'])} recommendations")
        
        # 6. 格式化返回结果
        recommendations = []
        for rec in result['recommendations']:
            # 获取共同主题的可读名称
            common_topics = []
            for label_id in rec['common_labels'][:10]:  # 最多显示 10 个
                topic_name = recommendation_system.explainer.gnd_mapping.get(
                    label_id, label_id
                )
                common_topics.append({
                    'id': label_id,
                    'name': topic_name[:80] if len(topic_name) > 80 else topic_name
                })
            
            # 真实主题（从 metadata）
            rec_subjects = []
            subject_ids = rec.get('subjects', [])
            for subject_id in subject_ids[:10]:  # 最多显示 10 个
                name = recommendation_system.explainer.gnd_mapping.get(
                    subject_id, subject_id
                )
                rec_subjects.append({
                    'id': subject_id,
                    'name': name[:80] if len(name) > 80 else name
                })
            
            # 预测主题（从 XGBoost）
            rec_predicted_subjects = []
            for subject_id in rec.get('predicted_subjects', [])[:10]:
                name = recommendation_system.explainer.gnd_mapping.get(
                    subject_id, subject_id
                )
                rec_predicted_subjects.append({
                    'id': subject_id,
                    'name': name[:80] if len(name) > 80 else name
                })
            
            recommendations.append({
                'rank': rec['rank'],
                'paper_id': rec['paper_id'],
                'title': rec['title'],
                'authors': rec['authors'],
                'year': rec['year'],
                'type': rec.get('type', 'Unknown'),
                'similarity': f"{rec['similarity']:.2%}",
                'cosine_similarity': f"{rec['cosine_similarity']:.2%}",
                'label_similarity': f"{rec['label_similarity']:.2%}",
                'common_topics_count': len(rec['common_labels']),
                'common_topics': common_topics,
                'subjects': rec_subjects,
                'predicted_subjects': rec_predicted_subjects,
                'explanation': rec['explanation']
            })
        
        return jsonify({
            'success': True,
            'query': {
                'title': title,
                'abstract': abstract,
                'predicted_subjects': query_predicted_subjects[:15]  # 最多显示 15 个
            },
            'recommendations': recommendations
        })
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"\n❌ Error occurred:")
        print(error_trace)
        
        return jsonify({
            'success': False,
            'error': str(e),
            'trace': error_trace
        }), 500


@app.route('/health')
def health():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': recommendation_system is not None,
        'embedder_loaded': embedder is not None,
        'llm_enabled': recommendation_system.explainer.use_llm if recommendation_system else False
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
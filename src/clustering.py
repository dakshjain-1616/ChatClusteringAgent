import numpy as np
import torch
import logging
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

LARGE_DATASET_THRESHOLD = 2000
EMBEDDING_BATCH_SIZE = 64

AI_ML_DOMAINS = [
    'Transformer Architecture Design and Tuning',
    'Data Augmentation Techniques (Image/Text/Audio)',
    'CUDA/GPU Memory Debugging and Optimization',
    'Loss Function Selection and Optimization',
    'API Integration and Deployment Errors',
    'Model Training Configuration (Batch/Epoch/LR)',
    'Gradient Descent Issues (Vanishing/Exploding)',
    'Hyperparameter Tuning (Grid/Random/Bayesian)',
    'Dataset Preprocessing and Tokenization',
    'Memory Management and VRAM Optimization',
    'Inference Speed Optimization and Quantization',
    'Distributed Training (DDP/FSDP/DeepSpeed)',
    'Mixed Precision Training (FP16/BF16)',
    'Transfer Learning and Fine-Tuning',
    'Overfitting and Regularization Techniques',
    'Learning Rate Scheduling Strategies',
    'Weight Initialization Methods',
    'Batch Normalization vs Layer Normalization',
    'Attention Mechanism Implementation',
    'Cross-Entropy vs Focal Loss',
    'Activation Function Selection (ReLU/GELU/Swish)',
    'Optimizer Selection (Adam/SGD/AdamW)',
    'Data Loading Bottlenecks',
    'Model Architecture Search (NAS)',
    'Pruning and Model Compression',
    'Knowledge Distillation',
    'Few-Shot Learning Approaches',
    'Zero-Shot Classification Tasks',
    'Multi-Task Learning Setup',
    'Contrastive Learning Methods',
    'Self-Supervised Learning Techniques',
    'Active Learning Strategies',
    'Curriculum Learning Implementation',
    'Domain Adaptation Techniques',
    'Adversarial Training',
    'GAN Training Stability',
    'VAE Latent Space Optimization',
    'Diffusion Model Configuration',
    'Explainability Tools (SHAP/LIME/GradCAM)',
    'MLOps Pipeline and CI/CD Integration',
    'Model Serving and REST API Development',
    'Retrieval-Augmented Generation (RAG) Setup',
    'Vector Database Integration (FAISS/Pinecone)',
    'Prompt Engineering and LLM Fine-tuning',
    'Tokenizer Configuration and Vocabulary',
    'Positional Encoding and Embeddings',
    'Beam Search and Decoding Strategies',
    'Evaluation Metrics for NLP (ROUGE/BERTScore)',
    'Image Classification Architecture Selection',
    'Object Detection Model Configuration (YOLO/RCNN)',
    'Semantic Segmentation Implementation',
    'Time Series Forecasting Models (LSTM/Prophet)',
    'Recommendation System Design (Collaborative/Content)',
    'Reinforcement Learning Policy Networks',
    'Q-Learning and Deep Q-Networks (DQN)',
    'Policy Gradient Methods (PPO/A3C)',
    'Multi-Agent Reinforcement Learning',
    'Imitation Learning from Demonstrations',
    'Reward Shaping Techniques',
    'Exploration vs Exploitation Strategies',
    'Speech Recognition Model Training',
    'Text-to-Speech Synthesis',
    'Audio Feature Extraction (MFCC/Spectrogram)',
    'Music Generation Models',
    'Video Action Recognition',
    ' 3D Point Cloud Processing',
    'Graph Neural Networks (GNN)',
    'Causal Inference in ML',
    'Fairness and Bias Mitigation',
    'Privacy-Preserving ML (Federated/Differential)',
    'Model Interpretability',
    'Uncertainty Quantification',
    'Anomaly Detection Algorithms',
    'Clustering and Dimensionality Reduction',
    'Ensemble Methods (Bagging/Boosting)',
    'AutoML and Hyperparameter Optimization',
    'Data Versioning and Experiment Tracking',
    'A/B Testing for ML Models',
    'Model Monitoring and Drift Detection',
    'Edge Deployment Optimization',
    'ONNX and Model Conversion',
    'TensorRT Optimization',
    'Core ML and Mobile Deployment',
    'WebAssembly ML Inference',
    'Ethical AI and Responsible ML',
    'Synthetic Data Generation',
    'Data Annotation and Labeling',
    'Cross-Validation Strategies',
    'Stratified Sampling Techniques',
    'Imbalanced Dataset Handling',
    'Missing Data Imputation',
    'Feature Engineering and Selection',
    'Categorical Encoding Methods',
    'Time Series Data Augmentation',
    'General AI/ML Discussion',
    'Debugging and Troubleshooting',
    'Best Practices and Code Review',
    'Research Paper Implementation',
    'Concept Explanation and Theory',
    'Tool and Library Recommendations'
]

class SemanticClusterer:
    def __init__(self, model_name: str = 'sentence-transformers/all-mpnet-base-v2'):
        logger.info(f"Loading sentence transformer model: {model_name}")
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        if self.device == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
        
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embeddings = None
        self.cluster_labels = None
        self.optimal_k = None
        
        logger.info("Loading zero-shot classification model: facebook/bart-large-mnli")
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if self.device == 'cuda' else -1
        )
        logger.info("Zero-shot classifier loaded successfully")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        n_texts = len(texts)
        logger.info(f"Generating embeddings for {n_texts} texts")
        
        if n_texts > LARGE_DATASET_THRESHOLD:
            logger.info(f"Large dataset detected (>{LARGE_DATASET_THRESHOLD}). Using batch processing with batch_size={EMBEDDING_BATCH_SIZE}")
            self.embeddings = self.model.encode(
                texts,
                batch_size=EMBEDDING_BATCH_SIZE,
                show_progress_bar=True,
                convert_to_numpy=True,
                device=self.device
            )
        else:
            logger.info(f"Processing with default batch size")
            self.embeddings = self.model.encode(
                texts,
                show_progress_bar=True,
                convert_to_numpy=True,
                device=self.device
            )
        
        logger.info(f"Generated embeddings with shape: {self.embeddings.shape}")
        
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        
        return self.embeddings
    
    def find_optimal_clusters(self, embeddings: np.ndarray, min_k: int = 2, max_k: int = 10) -> Tuple[int, str]:
        max_k = min(max_k, len(embeddings) - 1)
        if max_k < min_k:
            logger.warning(f"Dataset too small for clustering, using k=2")
            return 2, 'KMeans'
        
        k_range = range(min_k, max_k + 1)
        
        logger.info(f"Finding optimal clustering strategy (testing k={min_k} to {max_k})")
        logger.info("Dual-strategy evaluation: KMeans vs Agglomerative Clustering")
        
        best_score = -1
        best_k = min_k
        best_algorithm = 'KMeans'
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(embeddings)
            kmeans_score = silhouette_score(embeddings, kmeans_labels)
            
            agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
            agg_labels = agg.fit_predict(embeddings)
            agg_score = silhouette_score(embeddings, agg_labels)
            
            logger.info(f"k={k}: KMeans silhouette={kmeans_score:.4f}, Agglomerative silhouette={agg_score:.4f}")
            
            if kmeans_score > best_score:
                best_score = kmeans_score
                best_k = k
                best_algorithm = 'KMeans'
            
            if agg_score > best_score:
                best_score = agg_score
                best_k = k
                best_algorithm = 'AgglomerativeClustering'
        
        logger.info(f"Optimal configuration: {best_algorithm} with k={best_k} (silhouette={best_score:.4f})")
        return best_k, best_algorithm
    
    def cluster(self, texts: List[str], k: int = None) -> Tuple[np.ndarray, int]:
        embeddings = self.generate_embeddings(texts)
        n_samples = len(texts)
        
        if k is None:
            k, algorithm = self.find_optimal_clusters(embeddings)
        else:
            algorithm = 'KMeans'
        
        self.optimal_k = k
        
        if n_samples > LARGE_DATASET_THRESHOLD:
            logger.info(f"Large dataset ({n_samples} samples > {LARGE_DATASET_THRESHOLD}). Using MiniBatchKMeans")
            clusterer = MiniBatchKMeans(
                n_clusters=k,
                random_state=42,
                batch_size=256,
                n_init=10,
                max_iter=100
            )
            algorithm = 'MiniBatchKMeans'
        elif algorithm == 'KMeans':
            logger.info(f"Standard dataset ({n_samples} samples). Using KMeans")
            clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
        else:
            logger.info(f"Using {algorithm} for {n_samples} samples")
            clusterer = AgglomerativeClustering(n_clusters=k, linkage='ward')
        
        logger.info(f"Clustering with {algorithm}, k={k}")
        self.cluster_labels = clusterer.fit_predict(embeddings)
        
        return self.cluster_labels, k
    
    def classify_ai_ml_domain(self, text: str) -> Tuple[str, float]:
        try:
            result = self.classifier(text, AI_ML_DOMAINS, multi_label=False)
            domain = result['labels'][0]
            confidence = result['scores'][0]
            
            domain_short = domain.split('(')[0].strip() if '(' in domain else domain.split(' and ')[0]
            return domain_short, confidence
        except Exception as e:
            logger.warning(f"Classification failed: {e}")
            return "General AI/ML", 0.5
    
    def generate_cluster_labels(self, texts: List[str], cluster_assignments: np.ndarray, 
                               n_keywords: int = 5) -> Dict[int, Dict]:
        unique_clusters = np.unique(cluster_assignments)
        cluster_info = {}
        
        logger.info("Generating cluster labels using Semantic Intent Classification (facebook/bart-large-mnli)")
        
        for cluster_id in unique_clusters:
            cluster_texts = [texts[i] for i in range(len(texts)) if cluster_assignments[i] == cluster_id]
            cluster_size = len(cluster_texts)
            
            if cluster_size == 0:
                continue
            
            try:
                domain_votes = {}
                domain_confidences = {}
                
                sample_size = min(3, len(cluster_texts))
                sample_indices = np.random.choice(len(cluster_texts), sample_size, replace=False) if len(cluster_texts) > sample_size else range(len(cluster_texts))
                
                logger.info(f"Classifying cluster {cluster_id} with {sample_size} samples")
                
                for idx in sample_indices:
                    text = cluster_texts[idx]
                    domain, confidence = self.classify_ai_ml_domain(text)
                    
                    domain_votes[domain] = domain_votes.get(domain, 0) + 1
                    if domain not in domain_confidences:
                        domain_confidences[domain] = []
                    domain_confidences[domain].append(confidence)
                
                primary_domain = max(domain_votes.items(), key=lambda x: x[1])[0]
                avg_confidence = np.mean(domain_confidences[primary_domain])
                
                domain_distribution = {k: v/sample_size for k, v in domain_votes.items()}
                
                sample_texts = cluster_texts[:3] if len(cluster_texts) >= 3 else cluster_texts
                
                cluster_info[int(cluster_id)] = {
                    'size': cluster_size,
                    'domain': primary_domain,
                    'confidence': float(avg_confidence),
                    'domain_distribution': domain_distribution,
                    'label': primary_domain,
                    'sample_texts': sample_texts
                }
                
                logger.info(f"Cluster {cluster_id}: {primary_domain} (confidence: {avg_confidence:.3f}, n={cluster_size})")
                
            except Exception as e:
                logger.warning(f"Could not classify cluster {cluster_id}: {e}")
                cluster_info[int(cluster_id)] = {
                    'size': cluster_size,
                    'domain': 'Unclassified',
                    'confidence': 0.0,
                    'domain_distribution': {},
                    'label': f'Cluster {cluster_id}',
                    'sample_texts': cluster_texts[:3]
                }
        
        return cluster_info
    
    def cluster_with_metadata(self, data: List[Dict]) -> Dict:
        texts = [item['text'] for item in data]
        cluster_assignments, k = self.cluster(texts)
        cluster_info = self.generate_cluster_labels(texts, cluster_assignments)
        
        clustered_data = []
        for i, item in enumerate(data):
            clustered_item = item.copy()
            clustered_item['cluster_id'] = int(cluster_assignments[i])
            clustered_item['cluster_label'] = cluster_info[cluster_assignments[i]]['label']
            clustered_data.append(clustered_item)
        
        return {
            'data': clustered_data,
            'cluster_info': cluster_info,
            'num_clusters': k,
            'embeddings': self.embeddings
        }
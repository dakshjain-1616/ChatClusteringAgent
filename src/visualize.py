import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from typing import Dict, List
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClusterVisualizer:
    def __init__(self, output_dir: str = './output'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def reduce_dimensions(self, embeddings: np.ndarray, method: str = 'tsne') -> np.ndarray:
        logger.info(f"Reducing dimensions using {method.upper()}")
        
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        reduced = reducer.fit_transform(embeddings)
        return reduced
    
    def create_scatter_plot(self, embeddings: np.ndarray, labels: np.ndarray, 
                           cluster_info: Dict, title: str, filename: str):
        coords_2d = self.reduce_dimensions(embeddings, method='tsne')
        
        plt.figure(figsize=(14, 9))
        
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, cluster_id in enumerate(unique_labels):
            mask = labels == cluster_id
            info = cluster_info[cluster_id]
            domain = info.get('domain', info.get('label', f'Cluster {cluster_id}'))
            cluster_size = info['size']
            confidence = info.get('confidence', 0.0)
            
            label_text = f'{domain} (n={cluster_size}, conf={confidence:.2f})' if confidence > 0 else f'{domain} (n={cluster_size})'
            
            plt.scatter(coords_2d[mask, 0], coords_2d[mask, 1], 
                       c=[colors[i]], label=label_text,
                       alpha=0.6, edgecolors='w', s=100)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('t-SNE Component 1 (Semantic Space)', fontsize=12)
        plt.ylabel('t-SNE Component 2 (Semantic Space)', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.tight_layout()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved scatter plot to {output_path}")
        return output_path
    
    def create_pie_chart(self, cluster_info: Dict, title: str, filename: str):
        plt.figure(figsize=(12, 9))
        
        sizes = [info['size'] for info in cluster_info.values()]
        labels = []
        for info in cluster_info.values():
            domain = info.get('domain', info.get('label', 'Unknown'))
            size = info['size']
            confidence = info.get('confidence', 0.0)
            label_text = f"{domain}\n(n={size})" if confidence == 0 else f"{domain}\n(n={size}, {confidence:.1%})"
            labels.append(label_text)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_info)))
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 9})
        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('equal')
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved pie chart to {output_path}")
        return output_path
    
    def generate_text_report(self, first_chat_results: Dict, full_history_results: Dict,
                            filename: str = 'analysis_report.txt'):
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("AI/ML CHAT CLUSTERING ANALYSIS REPORT\n")
            f.write("Semantic Analysis with Domain Classification (facebook/bart-large-mnli)\n")
            f.write("="*80 + "\n\n")
            
            f.write("FIRST CHAT ANALYSIS - Initial User Intent\n")
            f.write("-"*80 + "\n")
            f.write(f"Total Conversations: {len(first_chat_results['data'])}\n")
            f.write(f"Number of Clusters: {first_chat_results['num_clusters']}\n\n")
            
            for cluster_id, info in sorted(first_chat_results['cluster_info'].items()):
                domain = info.get('domain', info.get('label', f'Cluster {cluster_id}'))
                f.write(f"\nCluster {cluster_id}: {domain}\n")
                f.write(f"  Size: {info['size']} conversations ({info['size']/len(first_chat_results['data'])*100:.1f}%)\n")
                f.write(f"  Confidence: {info.get('confidence', 0.0):.2%}\n")
                
                if 'domain_distribution' in info and info['domain_distribution']:
                    f.write(f"  Domain Distribution:\n")
                    for dom, ratio in sorted(info['domain_distribution'].items(), key=lambda x: x[1], reverse=True):
                        f.write(f"    - {dom}: {ratio:.1%}\n")
                
                f.write(f"  Sample Conversations:\n")
                for i, sample in enumerate(info['sample_texts'][:3], 1):
                    truncated = sample[:250] + '...' if len(sample) > 250 else sample
                    f.write(f"    {i}. {truncated}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("FULL HISTORY ANALYSIS - Complete Conversation Patterns\n")
            f.write("-"*80 + "\n")
            f.write(f"Total Conversations: {len(full_history_results['data'])}\n")
            f.write(f"Number of Clusters: {full_history_results['num_clusters']}\n\n")
            
            for cluster_id, info in sorted(full_history_results['cluster_info'].items()):
                domain = info.get('domain', info.get('label', f'Cluster {cluster_id}'))
                f.write(f"\nCluster {cluster_id}: {domain}\n")
                f.write(f"  Size: {info['size']} conversations ({info['size']/len(full_history_results['data'])*100:.1f}%)\n")
                f.write(f"  Confidence: {info.get('confidence', 0.0):.2%}\n")
                
                if 'domain_distribution' in info and info['domain_distribution']:
                    f.write(f"  Domain Distribution:\n")
                    for dom, ratio in sorted(info['domain_distribution'].items(), key=lambda x: x[1], reverse=True):
                        f.write(f"    - {dom}: {ratio:.1%}\n")
                
                f.write(f"  Sample Histories:\n")
                for i, sample in enumerate(info['sample_texts'][:2], 1):
                    truncated = sample[:250] + '...' if len(sample) > 250 else sample
                    f.write(f"    {i}. {truncated}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("DEEP SEMANTIC INSIGHTS - AI/ML Problem Analysis\n")
            f.write("-"*80 + "\n\n")
            
            f.write("First Chat Intent Patterns:\n")
            first_domains = {}
            for info in first_chat_results['cluster_info'].values():
                domain = info.get('domain', 'Unknown')
                first_domains[domain] = first_domains.get(domain, 0) + info['size']
            
            for domain, count in sorted(first_domains.items(), key=lambda x: x[1], reverse=True):
                pct = (count / len(first_chat_results['data'])) * 100
                f.write(f"  - {domain}: {count} chats ({pct:.1f}%)\n")
            
            f.write("\nFull History Conversation Patterns:\n")
            full_domains = {}
            for info in full_history_results['cluster_info'].values():
                domain = info.get('domain', 'Unknown')
                full_domains[domain] = full_domains.get(domain, 0) + info['size']
            
            for domain, count in sorted(full_domains.items(), key=lambda x: x[1], reverse=True):
                pct = (count / len(full_history_results['data'])) * 100
                f.write(f"  - {domain}: {count} chats ({pct:.1f}%)\n")
            
            f.write("\nKey Observations:\n")
            f.write(f"  - First chats show {first_chat_results['num_clusters']} distinct AI/ML problem types\n")
            f.write(f"  - Complete histories reveal {full_history_results['num_clusters']} conversation patterns\n")
            
            first_domains_set = set(first_domains.keys())
            full_domains_set = set(full_domains.keys())
            common_domains = first_domains_set.intersection(full_domains_set)
            
            if common_domains:
                f.write(f"\n  Consistent AI/ML domains across both analyses:\n")
                for domain in sorted(common_domains):
                    first_count = first_domains.get(domain, 0)
                    full_count = full_domains.get(domain, 0)
                    f.write(f"    - {domain}: {first_count} → {full_count} chats (first → full)\n")
            
            new_in_full = full_domains_set - first_domains_set
            if new_in_full:
                f.write(f"\n  Domains emerging in full conversation history:\n")
                for domain in sorted(new_in_full):
                    f.write(f"    - {domain}: {full_domains[domain]} chats\n")
            
            f.write("\n" + "="*80 + "\n")
        
        logger.info(f"Generated analysis report: {output_path}")
        return output_path
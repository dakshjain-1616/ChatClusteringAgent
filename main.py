import json
import sys
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_loader import ChatDataLoader
from clustering import SemanticClusterer
from visualize import ClusterVisualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Chat Clustering Tool - Semantic analysis of user conversations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --input custom_chats.json --output results/
  python main.py -i data/chats.json -o output/analysis/
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='last_50_chats.json',
        help='Path to input JSON file containing chat data (default: last_50_chats.json)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output',
        help='Output directory for results (default: output/)'
    )
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    logger.info("="*80)
    logger.info("CHAT CLUSTERING TOOL - Starting Pipeline")
    logger.info("="*80)
    
    project_dir = Path(__file__).parent
    
    input_file = Path(args.input)
    if not input_file.is_absolute():
        input_file = project_dir / input_file
    
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = project_dir / output_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output directory: {output_dir}")
    
    if not input_file.exists():
        logger.error(f"Error: Input file '{input_file}' not found!")
        sys.exit(1)
    
    loader = ChatDataLoader(input_file)
    
    logger.info("\n" + "="*80)
    logger.info("STEP 1: Loading and Parsing Data")
    logger.info("="*80)
    
    stats = loader.get_statistics()
    logger.info(f"Dataset Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    first_chat_data, full_history_data = loader.parse_conversations()
    
    logger.info("\n" + "="*80)
    logger.info("STEP 2: Clustering First Chats")
    logger.info("="*80)
    
    first_chat_clusterer = SemanticClusterer()
    first_chat_results = first_chat_clusterer.cluster_with_metadata(first_chat_data)
    
    logger.info(f"\nFirst Chat Clustering Results:")
    logger.info(f"  Number of clusters: {first_chat_results['num_clusters']}")
    for cluster_id, info in first_chat_results['cluster_info'].items():
        logger.info(f"  Cluster {cluster_id}: {info['label']} (n={info['size']})")
    
    logger.info("\n" + "="*80)
    logger.info("STEP 3: Clustering Full Histories")
    logger.info("="*80)
    
    full_history_clusterer = SemanticClusterer()
    full_history_results = full_history_clusterer.cluster_with_metadata(full_history_data)
    
    logger.info(f"\nFull History Clustering Results:")
    logger.info(f"  Number of clusters: {full_history_results['num_clusters']}")
    for cluster_id, info in full_history_results['cluster_info'].items():
        logger.info(f"  Cluster {cluster_id}: {info['label']} (n={info['size']})")
    
    logger.info("\n" + "="*80)
    logger.info("STEP 4: Generating Visualizations")
    logger.info("="*80)
    
    visualizer = ClusterVisualizer(output_dir=output_dir)
    
    first_labels = [item['cluster_id'] for item in first_chat_results['data']]
    full_labels = [item['cluster_id'] for item in full_history_results['data']]
    
    visualizer.create_scatter_plot(
        first_chat_results['embeddings'],
        first_labels,
        first_chat_results['cluster_info'],
        'First Chat Clusters - Semantic Analysis',
        'first_chat_scatter.png'
    )
    
    visualizer.create_pie_chart(
        first_chat_results['cluster_info'],
        'First Chat Distribution',
        'first_chat_pie.png'
    )
    
    visualizer.create_scatter_plot(
        full_history_results['embeddings'],
        full_labels,
        full_history_results['cluster_info'],
        'Full History Clusters - Semantic Analysis',
        'full_history_scatter.png'
    )
    
    visualizer.create_pie_chart(
        full_history_results['cluster_info'],
        'Full History Distribution',
        'full_history_pie.png'
    )
    
    logger.info("\n" + "="*80)
    logger.info("STEP 5: Generating Analysis Report")
    logger.info("="*80)
    
    report_path = visualizer.generate_text_report(
        first_chat_results,
        full_history_results,
        'analysis_report.txt'
    )
    
    logger.info("\n" + "="*80)
    logger.info("STEP 6: Saving Clustered Data")
    logger.info("="*80)
    
    output_data = {
        'first_chat_analysis': {
            'num_clusters': first_chat_results['num_clusters'],
            'cluster_info': first_chat_results['cluster_info'],
            'conversations': first_chat_results['data']
        },
        'full_history_analysis': {
            'num_clusters': full_history_results['num_clusters'],
            'cluster_info': full_history_results['cluster_info'],
            'conversations': full_history_results['data']
        },
        'metadata': {
            'total_conversations': len(first_chat_data),
            'embedding_model': 'all-MiniLM-L6-v2',
            'classification_model': 'facebook/bart-large-mnli',
            'clustering_algorithm': 'KMeans with Silhouette Score optimization',
            'labeling_method': 'Zero-Shot Semantic Intent Classification (AI/ML Domain-Aware)'
        }
    }
    
    output_json_path = output_dir / 'clustered_data.json'
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved clustered data to {output_json_path}")
    
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*80)
    logger.info("\nGenerated Files:")
    logger.info(f"  - {output_dir / 'first_chat_scatter.png'}")
    logger.info(f"  - {output_dir / 'first_chat_pie.png'}")
    logger.info(f"  - {output_dir / 'full_history_scatter.png'}")
    logger.info(f"  - {output_dir / 'full_history_pie.png'}")
    logger.info(f"  - {output_dir / 'analysis_report.txt'}")
    logger.info(f"  - {output_dir / 'clustered_data.json'}")
    logger.info("="*80)

if __name__ == '__main__':
    main()
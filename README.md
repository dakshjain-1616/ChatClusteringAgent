# Chat Clustering Tool

An intelligent chat clustering system that automatically groups user conversations into semantically meaningful clusters based on content analysis.

## Overview

This tool analyzes chat conversation data using semantic embeddings and machine learning to identify patterns in user interactions. It supports two analysis modes:
- **First Chat Analysis**: Analyzes initial user messages to understand entry intents
- **Full History Analysis**: Analyzes complete conversation threads to identify overall patterns

## Features

- Semantic-aware clustering using sentence-transformers (all-MiniLM-L6-v2)
- Automatic cluster detection using Silhouette Score optimization
- TF-IDF-based cluster labeling for human-interpretable summaries
- t-SNE dimensionality reduction for visualization
- Multiple output formats: scatter plots, pie charts, text reports, and JSON data
- GPU-accelerated embedding generation (falls back to CPU)

## Project Structure

```
```
chatClusteringTool/
├── main.py                      # Main pipeline orchestration
├── src/
│   ├── data_loader.py          # JSON parsing and data extraction
│   ├── clustering.py           # Semantic clustering engine
│   └── visualize.py            # Visualization and reporting
├── output/                      # Generated results
│   ├── first_chat_scatter.png
│   ├── first_chat_pie.png
│   ├── full_history_scatter.png
│   ├── full_history_pie.png
│   ├── analysis_report.txt
│   └── clustered_data.json
└── last_50_chats.json          # Input data
```
```

## Installation

1. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install sentence-transformers scikit-learn matplotlib numpy pandas
```

## Usage

Run the complete pipeline:
```bash
python main.py
```

The tool will:
1. Load and parse JSON chat data
2. Generate semantic embeddings for all conversations
3. Cluster conversations using optimal k determination
4. Generate visualizations and reports
5. Save all outputs to the `output/` directory

## Input Format

The tool expects JSON files with the following structure:
```json
[
  {
    "thread_id": "unique-id",
    "user_id": "user-id",
    "messages": [
      {
        "sender": "User",
        "content": "message text",
        "message_id": "msg-id",
        "created_at": "timestamp"
      }
    ]
  }
]
```

## Output Files

- **first_chat_scatter.png**: t-SNE visualization of first chat clusters
- **first_chat_pie.png**: Distribution pie chart for first chats
- **full_history_scatter.png**: t-SNE visualization of full history clusters
- **full_history_pie.png**: Distribution pie chart for full histories
- **analysis_report.txt**: Detailed text report with cluster summaries and insights
- **clustered_data.json**: Complete dataset with cluster assignments

## Technical Details

### Clustering Algorithm
- Model: sentence-transformers/all-MiniLM-L6-v2 (384-dimensional embeddings)
- Clustering: KMeans with Silhouette Score optimization (k=2 to 10)
- Labeling: TF-IDF keyword extraction with n-grams

### Visualization
- Dimensionality reduction: t-SNE (2D projection)
- Color coding: Distinct colors per cluster
- Interactive legends with cluster sizes

## Performance

- Processes 50 conversations in ~11 seconds on Tesla V100 GPU
- Scales efficiently to thousands of conversations
- Memory-optimized for large datasets

## Requirements

- Python 3.8+
- GPU recommended (CUDA-compatible) but CPU supported
- 2GB+ RAM for typical datasets
- ~500MB disk space for model cache

## License

This tool is provided as-is for chat analysis and clustering purposes.
# NEO Chat Topic Analysis Pipeline

A comprehensive tool for analyzing user chat messages to identify topics, sentiment patterns, and message types using the Kura framework and state-of-the-art NLP models.

## Features

- **Topic Clustering**: Automatically groups conversations into thematic clusters using sentence embeddings
- **Sentiment Analysis**: Evaluates user satisfaction through message-level sentiment scoring
- **Message Categorization**: Classifies messages into types (questions, commands, statements, feedback)
- **GPU Acceleration**: Supports both CPU and GPU computation with automatic hardware detection
- **Flexible Configuration**: Command-line arguments for complete customization
- **Rich Reporting**: Generates comprehensive Markdown reports and JSON data exports

## Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for acceleration)
- 4GB+ RAM (8GB+ recommended for large datasets)

## Installation

### 1. Clone or navigate to the project directory

```bash
cd /path/to/chatClusteringTool
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**Note**: The first run will download pre-trained models (~500MB). Ensure stable internet connection.

## Usage

### Basic Usage

Run analysis with default settings (10 clusters, auto-detect hardware):

```bash
python src/analyze_chats.py --input_path ./last_50_chats.json
```

### Advanced Usage

Customize all parameters:

```bash
python src/analyze_chats.py \
  --input_path ./data/my_chats.json \
  --output_dir ./custom_results \
  --num_clusters 15 \
  --top_keywords 10 \
  --device cuda
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input_path` | string | **Required** | Path to input JSON file with chat data |
| `--output_dir` | string | `./results` | Directory to save analysis outputs |
| `--num_clusters` | integer | `10` | Number of topic clusters to identify |
| `--top_keywords` | integer | `5` | Keywords to extract per cluster |
| `--device` | choice | `auto` | Computation device: `cuda`, `cpu`, or `auto` |

### Device Selection

- **`auto`** (default): Automatically detects GPU availability and uses best option
- **`cuda`**: Forces GPU usage (falls back to CPU if unavailable)
- **`cpu`**: Forces CPU-only computation (useful for debugging or small datasets)

## Input Data Format

The script expects a JSON file with the following structure:

```json
[
  {
    "thread_id": "unique_thread_id",
    "user_id": "user_identifier",
    "created_at": {"$date": "2026-01-15T10:30:00.000Z"},
    "messages": [
      {
        "sender": "User",
        "content": "How do I implement authentication?",
        "created_at": "2026-01-15T10:30:00.000Z"
      },
      {
        "sender": "Assistant",
        "content": "Here's how to implement authentication..."
      }
    ],
    "cycles_consumed": 5
  }
]
```

**Key Fields**:
- `messages[].sender`: Must include `"User"` messages for analysis
- `messages[].content`: Text content to analyze
- `thread_id` or `_id.$oid`: Unique conversation identifier

## Output Files

Analysis generates two files in the output directory:

### 1. JSON Results (`analysis_results_YYYYMMDD_HHMMSS.json`)

Complete structured data including:
- Topic clusters with conversation assignments
- Cluster keywords and statistics
- Message type distributions
- User satisfaction scores
- Conversation-level sentiment breakdowns
- Analysis metadata (models used, parameters)

### 2. Markdown Report (`topic_analysis_report_YYYYMMDD_HHMMSS.md`)

Human-readable report with:
- Executive summary with key metrics
- User satisfaction analysis (high/medium/low breakdown)
- Detailed topic cluster descriptions
- Message type distribution charts
- Top satisfied/dissatisfied conversations
- Methodology documentation

## Environment-Specific Setup

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/analyze_chats.py --input_path ./data.json
```

### Windows (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src\analyze_chats.py --input_path .\data.json
```

### Docker

```bash
docker run -v $(pwd):/workspace -w /workspace python:3.10 bash -c \
  "pip install -r requirements.txt && python src/analyze_chats.py --input_path /workspace/data.json"
```

### Google Colab

```python
!git clone <repository_url>
%cd chatClusteringTool
!pip install -r requirements.txt
!python src/analyze_chats.py --input_path ./last_50_chats.json --device cuda
```

## Performance Considerations

### GPU vs CPU

| Dataset Size | GPU Time | CPU Time | Recommended |
|--------------|----------|----------|-------------|
| 50 chats | ~30s | ~2min | Either |
| 500 chats | ~2min | ~15min | GPU |
| 5000+ chats | ~10min | ~2hrs | GPU |

### Memory Requirements

- **Small (< 100 chats)**: 2GB RAM
- **Medium (100-1000 chats)**: 4-8GB RAM
- **Large (1000+ chats)**: 8-16GB RAM
- **GPU VRAM**: 4GB minimum, 8GB+ recommended

## Troubleshooting

### "No module named 'kura'"

Ensure Kura is installed or the `kura/` directory is in the project root:

```bash
pip install kura
# OR if using local Kura
export PYTHONPATH="${PYTHONPATH}:$(pwd)/kura"
```

### "CUDA out of memory"

Reduce cluster count or force CPU usage:

```bash
python src/analyze_chats.py --input_path data.json --num_clusters 5 --device cpu
```

### "File not found" errors

Use absolute paths or ensure working directory is correct:

```bash
python src/analyze_chats.py --input_path /full/path/to/data.json
```

## Examples

### Analyzing Different Datasets

```bash
# Small dataset with fewer clusters
python src/analyze_chats.py --input_path ./small_sample.json --num_clusters 5

# Large dataset with detailed keywords
python src/analyze_chats.py --input_path ./large_dataset.json --num_clusters 20 --top_keywords 15

# Testing on CPU for reproducibility
python src/analyze_chats.py --input_path ./test_data.json --device cpu
```

### Integration with Other Tools

```bash
# Export results to specific location
python src/analyze_chats.py \
  --input_path ./chats.json \
  --output_dir /shared/analysis_$(date +%Y%m%d)

# Process and upload results
python src/analyze_chats.py --input_path ./data.json --output_dir ./temp
aws s3 sync ./temp s3://mybucket/analysis/
```

## Kura Framework Integration

This tool leverages the [Kura framework](https://github.com/567-labs/kura) for conversation data structures and processing. Kura provides:

- Type-safe `Conversation` and `Message` classes
- Checkpoint management for large datasets
- Standardized conversation formats
- Extensible processing pipelines

## Models Used

- **Sentence Embeddings**: `all-MiniLM-L6-v2` (384-dim, fast inference)
- **Sentiment Analysis**: `distilbert-base-uncased-finetuned-sst-2-english`
- **Clustering**: K-Means with optimized initialization

## License

[Specify your license here]

## Contributing

[Contribution guidelines here]

## Contact

For questions or issues, please contact [your contact info] or open an issue on GitHub.
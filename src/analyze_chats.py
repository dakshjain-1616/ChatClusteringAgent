import json
import os
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from datetime import datetime
from transformers import pipeline
from rich.console import Console
from dataclasses import dataclass, field

console = Console()

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Message:
    created_at: datetime
    role: str
    content: str

@dataclass
class Conversation:
    chat_id: str
    created_at: datetime
    messages: List[Message]
    metadata: Dict[str, Any] = field(default_factory=dict)

# ---------------------------------------------------------------------------
# 1. Load chat data
# ---------------------------------------------------------------------------

def load_chat_data(data_file: Path) -> List[Conversation]:
    console.print(f"[bold blue]Loading chat data from {data_file}...[/bold blue]")
    if not data_file.exists():
        console.print(f"[bold red]Error: Data file not found at {data_file}[/bold red]")
        sys.exit(1)
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except json.JSONDecodeError as e:
        console.print(f"[bold red]Error: Invalid JSON format in {data_file}: {e}[/bold red]")
        sys.exit(1)
    conversations: List[Conversation] = []
    for chat in raw_data:
        user_messages = [msg for msg in chat.get("messages", []) if msg.get("sender") == "User"]
        if not user_messages:
            continue
        messages: List[Message] = []
        for msg in user_messages:
            created_at_str = msg.get("created_at", chat.get("created_at", {}).get("$date", "2026-01-01T00:00:00.000Z"))
            if isinstance(created_at_str, dict):
                created_at_str = created_at_str.get("$date", "2026-01-01T00:00:00.000Z")
            created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
            messages.append(
                Message(
                    created_at=created_at,
                    role="user",
                    content=msg["content"],
                )
            )
        chat_created_at = chat.get("created_at", {})
        if isinstance(chat_created_at, dict):
            chat_created_at = chat_created_at.get("$date", "2026-01-01T00:00:00.000Z")
        chat_created_at = datetime.fromisoformat(chat_created_at.replace("Z", "+00:00"))
        conv = Conversation(
            chat_id=chat.get("thread_id", chat.get("_id", {}).get("$oid", "")),
            created_at=chat_created_at,
            messages=messages,
            metadata={
                "thread_id": chat.get("thread_id", ""),
                "user_id": chat.get("user_id", ""),
                "cycles_consumed": chat.get("cycles_consumed", 0),
                "total_messages": len(chat.get("messages", [])),
                "user_messages_count": len(user_messages),
            },
        )
        conversations.append(conv)
    console.print(f"[green]✓ Loaded {len(conversations)} conversations with user messages[/green]")
    return conversations

# ---------------------------------------------------------------------------
# 2. Topic extraction via embeddings
# ---------------------------------------------------------------------------

def extract_topics_with_embeddings(
    conversations: List[Conversation], n_clusters: int = 10, device: str | None = None
) -> Dict[str, Any]:
    console.print("\n[bold blue]Step 1: Extracting topics using sentence embeddings...[/bold blue]")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[cyan]Using device: {device}[/cyan]")
    if device == "cuda" and torch.cuda.is_available():
        console.print(f"[cyan]GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)[/cyan]")
    elif device == "cuda" and not torch.cuda.is_available():
        console.print("[yellow]⚠ GPU requested but not available, falling back to CPU[/yellow]")
        device = "cpu"
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    conversation_texts = [" ".join([msg.content for msg in conv.messages]) for conv in conversations]
    conversation_ids = [conv.chat_id for conv in conversations]
    console.print(f"[cyan]Generating embeddings for {len(conversation_texts)} conversations...[/cyan]")
    embeddings = model.encode(conversation_texts, show_progress_bar=True, convert_to_numpy=True)
    console.print(f"[cyan]Clustering conversations into {n_clusters} topics...[/cyan]")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    clusters: Dict[int, List[Dict[str, str]]] = {}
    for idx, label in enumerate(cluster_labels):
        clusters.setdefault(label, []).append(
            {"chat_id": conversation_ids[idx], "text": conversation_texts[idx][:200]}
        )
    console.print("[green]✓ Topic clustering complete[/green]")
    return {
        "clusters": clusters,
        "embeddings": embeddings,
        "cluster_labels": cluster_labels,
    }

# ---------------------------------------------------------------------------
# 3. Keyword extraction per cluster
# ---------------------------------------------------------------------------

def extract_cluster_keywords(clusters: Dict[int, List[Dict[str, str]]], top_n: int = 5) -> Dict[int, List[str]]:
    console.print("\n[bold blue]Step 2: Extracting keywords for each cluster...[/bold blue]")
    cluster_keywords: Dict[int, List[str]] = {}
    for cluster_id, convs in clusters.items():
        all_text = " ".join([conv["text"] for conv in convs])
        try:
            vectorizer = TfidfVectorizer(max_features=100, stop_words="english", ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform([all_text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            top_indices = scores.argsort()[-top_n:][::-1]
            keywords = [feature_names[i] for i in top_indices]
            cluster_keywords[cluster_id] = keywords
        except Exception:
            cluster_keywords[cluster_id] = ["general_conversation"]
    console.print("[green]✓ Keyword extraction complete[/green]")
    return cluster_keywords

# ---------------------------------------------------------------------------
# 4. Sentiment analysis for user satisfaction
# ---------------------------------------------------------------------------

def analyze_user_satisfaction(conversations: List[Conversation], device: str | None = None) -> Dict[str, Any]:
    console.print("\n[bold blue]Step 3: Analyzing user satisfaction with sentiment analysis...[/bold blue]")
    if device == "cuda" and torch.cuda.is_available():
        sentiment_device = 0
    else:
        sentiment_device = -1
    console.print(f"[cyan]Initializing sentiment analysis pipeline (device={'GPU' if sentiment_device == 0 else 'CPU'})...[/cyan]")
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=sentiment_device,
    )
    conversation_satisfaction: List[Dict[str, Any]] = []
    overall_scores: List[float] = []
    sentiment_distribution: Dict[str, int] = {"POSITIVE": 0, "NEGATIVE": 0}
    console.print(f"[cyan]Analyzing sentiment for {len(conversations)} conversations...[/cyan]")
    for conv in conversations:
        message_scores: List[float] = []
        message_sentiments: List[str] = []
        for msg in conv.messages:
            try:
                text = msg.content[:512]
                result = sentiment_analyzer(text)[0]
                sentiment = result["label"]
                score = result["score"]
                sentiment_distribution[sentiment] += 1
                normalized_score = score if sentiment == "POSITIVE" else (1 - score)
                message_scores.append(normalized_score)
                message_sentiments.append(sentiment)
            except Exception as e:
                console.print(f"[yellow]Warning: Could not analyze message: {str(e)[:50]}[/yellow]")
                message_scores.append(0.5)
                message_sentiments.append("NEUTRAL")
        if message_scores:
            avg_satisfaction = float(np.mean(message_scores))
            conversation_satisfaction.append(
                {
                    "chat_id": conv.chat_id,
                    "satisfaction_score": avg_satisfaction,
                    "message_count": len(message_scores),
                    "positive_messages": message_sentiments.count("POSITIVE"),
                    "negative_messages": message_sentiments.count("NEGATIVE"),
                }
            )
            overall_scores.append(avg_satisfaction)
    user_satisfaction_index = float(np.mean(overall_scores)) if overall_scores else 0.5
    satisfaction_levels = {
        "high_satisfaction": sum(1 for s in overall_scores if s >= 0.7),
        "medium_satisfaction": sum(1 for s in overall_scores if 0.4 <= s < 0.7),
        "low_satisfaction": sum(1 for s in overall_scores if s < 0.4),
    }
    console.print(f"[green]✓ User Satisfaction Index: {user_satisfaction_index:.2%}[/green]")
    console.print(f"[green]  High satisfaction conversations: {satisfaction_levels['high_satisfaction']}[/green]")
    console.print(f"[green]  Medium satisfaction conversations: {satisfaction_levels['medium_satisfaction']}[/green]")
    console.print(f"[green]  Low satisfaction conversations: {satisfaction_levels['low_satisfaction']}[/green]")
    return {
        "user_satisfaction_index": user_satisfaction_index,
        "conversation_satisfaction": conversation_satisfaction,
        "satisfaction_levels": satisfaction_levels,
        "sentiment_distribution": sentiment_distribution,
        "total_messages_analyzed": sum(sentiment_distribution.values()),
    }

# ---------------------------------------------------------------------------
# 5. Message type categorisation
# ---------------------------------------------------------------------------

def categorize_message_types(conversations: List[Conversation]) -> Dict[str, Any]:
    console.print("\n[bold blue]Step 4: Analyzing message types...[/bold blue]")
    message_types = {
        "questions": 0,
        "commands": 0,
        "requests": 0,
        "feedback": 0,
        "general": 0,
    }
    total_messages = 0
    for conv in conversations:
        for msg in conv.messages:
            total_messages += 1
            msg_lower = msg.content.lower()
            if any(q in msg_lower for q in ["?", "how", "what", "why", "when", "where", "who"]):
                message_types["questions"] += 1
            elif any(cmd in msg_lower for cmd in ["create", "generate", "make", "build", "train", "implement", "develop"]):
                message_types["commands"] += 1
            elif any(req in msg_lower for req in ["please", "need", "want", "help", "can you"]):
                message_types["requests"] += 1
            elif any(fb in msg_lower for fb in ["thanks", "good", "bad", "error", "issue", "problem"]):
                message_types["feedback"] += 1
            else:
                message_types["general"] += 1
    percentages = {
        k: (v / total_messages * 100) if total_messages > 0 else 0 for k, v in message_types.items()
    }
    console.print("[green]✓ Message type analysis complete[/green]")
    return {
        "counts": message_types,
        "percentages": percentages,
        "total_messages": total_messages,
    }

# ---------------------------------------------------------------------------
# 6. Persist results
# ---------------------------------------------------------------------------

def save_results(
    cluster_data: Dict[str, Any],
    cluster_keywords: Dict[int, List[str]],
    message_types: Dict[str, Any],
    satisfaction_data: Dict[str, Any],
    conversations: List[Conversation],
    results_dir: Path,
):
    console.print(f"\n[bold blue]Step 5: Saving results to {results_dir}...[/bold blue]")
    os.makedirs(results_dir, exist_ok=True)
    output_dir = results_dir
    cluster_summary: Dict[str, Any] = {}
    for cluster_id, convs in cluster_data["clusters"].items():
        cluster_summary[f"topic_{cluster_id}"] = {
            "conversation_count": len(convs),
            "percentage": (len(convs) / len(conversations)) * 100,
            "keywords": cluster_keywords.get(cluster_id, []),
            "sample_conversations": [c["chat_id"] for c in convs[:3]],
        }
    results = {
        "analysis_metadata": {
            "total_conversations": len(conversations),
            "total_user_messages": message_types["total_messages"],
            "analysis_timestamp": datetime.now().isoformat(),
            "clustering_method": "KMeans with SentenceTransformer embeddings",
            "embedding_model": "all-MiniLM-L6-v2",
        },
        "user_satisfaction": satisfaction_data,
        "topic_clusters": cluster_summary,
        "message_types": message_types,
        "detailed_clusters": {
            str(k): [
                {"chat_id": c["chat_id"], "preview": c["text"][:150]} for c in v
            ]
            for k, v in cluster_data["clusters"].items()
        },
    }
    json_file = results_dir / f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    console.print(f"[green]✓ Analysis results saved to {json_file}[/green]")
    generate_report(results, conversations, output_dir)
    return results

# ---------------------------------------------------------------------------
# 7. Markdown report generation
# ---------------------------------------------------------------------------

def generate_report(results: Dict[str, Any], conversations: List[Conversation], output_dir: Path) -> None:
    console.print("[cyan]Generating markdown report...[/cyan]")
    report_file = output_dir / f"topic_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# NEO User Chat Analysis Report\n\n")
        f.write(f"**Generated:** {results['analysis_metadata']['analysis_timestamp']}\n\n")
        f.write("---\n\n")
        f.write("## Executive Summary\n\n")
        f.write(f"- **Total Conversations Analyzed:** {results['analysis_metadata']['total_conversations']}\n")
        f.write(f"- **Total User Messages:** {results['analysis_metadata']['total_user_messages']}\n")
        f.write(f"- **User Satisfaction Index:** {results['user_satisfaction']['user_satisfaction_index']:.2%}\n")
        f.write(f"- **Topics Identified:** {len(results['topic_clusters'])}\n\n")
        f.write("## User Satisfaction Analysis\n\n")
        satisfaction = results["user_satisfaction"]
        f.write(f"### Overall User Satisfaction Index: {satisfaction['user_satisfaction_index']:.2%}\n\n")
        f.write("#### Satisfaction Distribution\n\n")
        levels = satisfaction["satisfaction_levels"]
        total_convs = sum(levels.values()) or 1
        f.write(f"- **High Satisfaction** (≥70%): {levels['high_satisfaction']} conversations ({levels['high_satisfaction']/total_convs*100:.1f}%)\n")
        f.write(f"- **Medium Satisfaction** (40-70%): {levels['medium_satisfaction']} conversations ({levels['medium_satisfaction']/total_convs*100:.1f}%)\n")
        f.write(f"- **Low Satisfaction** (<40%): {levels['low_satisfaction']} conversations ({levels['low_satisfaction']/total_convs*100:.1f}%)\n\n")
        f.write("#### Sentiment Distribution\n\n")
        sent_dist = satisfaction["sentiment_distribution"]
        total_msgs = satisfaction["total_messages_analyzed"] or 1
        f.write(f"- **Positive Messages:** {sent_dist['POSITIVE']} ({sent_dist['POSITIVE']/total_msgs*100:.1f}%)\n")
        f.write(f"- **Negative Messages:** {sent_dist['NEGATIVE']} ({sent_dist['NEGATIVE']/total_msgs*100:.1f}%)\n\n")
        f.write("#### Top 5 Most Satisfied Conversations\n\n")
        sorted_convs = sorted(
            satisfaction["conversation_satisfaction"], key=lambda x: x["satisfaction_score"], reverse=True
        )[:5]
        for i, conv in enumerate(sorted_convs, 1):
            f.write(f"{i}. Chat ID: `{conv['chat_id'][:16]}...` - Score: {conv['satisfaction_score']:.2%} ({conv['positive_messages']} positive, {conv['negative_messages']} negative)\n")
        f.write("\n")
        f.write("#### Top 5 Least Satisfied Conversations\n\n")
        sorted_convs_low = sorted(
            satisfaction["conversation_satisfaction"], key=lambda x: x["satisfaction_score"]
        )[:5]
        for i, conv in enumerate(sorted_convs_low, 1):
            f.write(f"{i}. Chat ID: `{conv['chat_id'][:16]}...` - Score: {conv['satisfaction_score']:.2%} ({conv['positive_messages']} positive, {conv['negative_messages']} negative)\n")
        f.write("\n---\n\n")
        f.write("## Topic Clusters\n\n")
        sorted_topics = sorted(
            results["topic_clusters"].items(), key=lambda x: x[1]["conversation_count"], reverse=True
        )
        for topic_id, topic_data in sorted_topics:
            f.write(f"### {topic_id.replace('_', ' ').title()}\n\n")
            f.write(f"- **Conversations:** {topic_data['conversation_count']} ({topic_data['percentage']:.1f}%)\n")
            f.write(f"- **Keywords:** {', '.join(topic_data['keywords'])}\n")
            f.write(f"- **Sample Chats:** {', '.join([f'`{c[:12]}...`' for c in topic_data['sample_conversations']])}\n\n")
        f.write("## Message Type Distribution\n\n")
        msg_types = results["message_types"]
        f.write(f"**Total Messages Analyzed:** {msg_types['total_messages']}\n\n")
        for msg_type, count in sorted(msg_types["counts"].items(), key=lambda x: x[1], reverse=True):
            percentage = msg_types["percentages"][msg_type]
            f.write(f"- **{msg_type.title()}:** {count} messages ({percentage:.1f}%)\n")
        f.write("\n---\n\n")
        f.write("## Methodology\n\n")
        f.write(f"- **Clustering Method:** {results['analysis_metadata']['clustering_method']}\n")
        f.write(f"- **Embedding Model:** {results['analysis_metadata']['embedding_model']}\n")
        f.write("- **Sentiment Model:** distilbert-base-uncased-finetuned-sst-2-english\n")
        f.write("- **Hardware:** GPU-accelerated (if available)\n")
    console.print(f"[green]✓ Report generated at {report_file}[/green]")

# ---------------------------------------------------------------------------
# 8. CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="NEO Chat Topic Analysis Pipeline - Analyze user chat messages for topics, sentiment, and message types",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic usage:
    python analyze_chats.py --input_path ./last_50_chats.json

  Custom output directory and cluster count:
    python analyze_chats.py --input_path ./chats.json --output_dir ./my_results --num_clusters 15

  Force CPU usage:
    python analyze_chats.py --input_path ./chats.json --device cpu

  Full customization:
    python analyze_chats.py --input_path ./data/chats.json --output_dir ./results --num_clusters 8 --top_keywords 10 --device cuda
"""
    )
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input JSON file containing chat data")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save analysis results (default: ./results)")
    parser.add_argument("--num_clusters", type=int, default=10, help="Number of topic clusters to identify (default: 10)")
    parser.add_argument("--top_keywords", type=int, default=5, help="Number of top keywords to extract per cluster (default: 5)")
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device to use for computation: cuda (GPU), cpu, or auto (default - detects available hardware)",
    )
    args = parser.parse_args()
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    console.print("\n[bold cyan]" + "=" * 60 + "[/bold cyan]")
    console.print("[bold cyan]NEO Chat Topic Analysis Pipeline (with User Satisfaction)[/bold cyan]")
    console.print("\n[cyan]Configuration:[/cyan]")
    console.print(f"  Input: {input_path}")
    console.print(f"  Output: {output_dir}")
    console.print(f"  Clusters: {args.num_clusters}")
    console.print(f"  Keywords per cluster: {args.top_keywords}")
    console.print(f"  Device: {device}")
    if device == "cuda" and not torch.cuda.is_available():
        console.print("[yellow]⚠ GPU requested but not available, falling back to CPU[/yellow]")
        device = "cpu"
    conversations = load_chat_data(input_path)
    cluster_data = extract_topics_with_embeddings(conversations, n_clusters=args.num_clusters, device=device)
    cluster_keywords = extract_cluster_keywords(cluster_data["clusters"], top_n=args.top_keywords)
    satisfaction_data = analyze_user_satisfaction(conversations, device=device)
    message_types = categorize_message_types(conversations)
    results = save_results(
        cluster_data, cluster_keywords, message_types, satisfaction_data, conversations, output_dir
    )
    console.print("\n[bold green]" + "=" * 60 + "[/bold green]")
    console.print("[bold green]Analysis Complete![/bold green]")
    console.print(f"[bold green]Topics Identified: {len(cluster_data['clusters'])}[/bold green]")
    console.print(f"[bold green]Messages Analyzed: {message_types['total_messages']}[/bold green]")
    console.print(f"[bold green]User Satisfaction Index: {satisfaction_data['user_satisfaction_index']:.2%}[/bold green]")
    console.print("[bold green]" + "=" * 60 + "[/bold green]")

if __name__ == "__main__":
    main()

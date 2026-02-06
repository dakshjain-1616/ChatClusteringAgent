cd /root/chatClusteringTool && ls -la
cd /root/chatClusteringTool && echo "=== Checking for any remaining test artifacts ==="
cd /root/chatClusteringTool && find . -name "__pycache__" -o -name "output*" -o -name "final_verification" -o -name "large_dataset_*.json" 2>/dev/null | grep -v "./venv" || echo "No test artifacts found outside venv"
cd /root/chatClusteringTool && echo "=== Verifying .gitignore exists and contains required patterns ==="
cd /root/chatClusteringTool && grep -E "(output/|output_\*/|__pycache__|\.pyc|large_dataset_5000\.json)" .gitignore
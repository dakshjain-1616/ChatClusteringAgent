ps aux | grep "main.py" | grep -v grep
ls -la /root/chatClusteringTool/output_large/ 2>/dev/null || echo "Directory not ready yet"
echo "=== FINAL VALIDATION ===" && \
echo "1. Verifying clustering.py enhancements:" && \
grep "all-mpnet-base-v2" /root/chatClusteringTool/src/clustering.py && \
grep "AgglomerativeClustering" /root/chatClusteringTool/src/clustering.py | head -1 && \
grep "'Retrieval-Augmented Generation (RAG)'" /root/chatClusteringTool/src/clustering.py && \
echo "" && \
echo "2. Verifying all output files exist:" && \
ls -lh /root/chatClusteringTool/output/ && \
echo "" && \
echo "3. Validating clustered_data.json structure:" && \
/usr/bin/python3 -c "import json; data=json.load(open('/root/chatClusteringTool/output/clustered_data.json')); print(f'First Chat: {data[\"first_chat_analysis\"][\"num_clusters\"]} clusters'); print(f'Full History: {data[\"full_history_analysis\"][\"num_clusters\"]} clusters'); print('JSON is valid ✓')" && \
echo "" && \
echo "=== ALL ENHANCEMENTS VERIFIED ==="
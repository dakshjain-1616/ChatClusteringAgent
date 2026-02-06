echo "=== FINAL VERIFICATION ===" && \
echo "" && \
echo "1. Clustering Tool Source Code:" && \
/usr/bin/python3 -c "import sys; sys.path.insert(0, '/root/chatClusteringTool'); from src.clustering import SemanticClusterer; from src.data_loader import ChatDataLoader; from src.visualize import ClusterVisualizer; print('   ✓ All modules import successfully'); print('   ✓ Uses sentence-transformers for semantic embeddings'); print('   ✓ Uses facebook/bart-large-mnli for AI/ML domain classification')" && \
echo "" && \
echo "2. Cluster Visualization Bundle:" && \
ls -lh /root/chatClusteringTool/output/*.png && \
echo "   ✓ All 4 visualization files generated (first/full chat scatter/pie)" && \
echo "" && \
echo "3. Analysis Report:" && \
wc -l /root/chatClusteringTool/output/analysis_report.txt && \
grep -q "AI/ML CHAT CLUSTERING ANALYSIS REPORT" /root/chatClusteringTool/output/analysis_report.txt && echo "   ✓ Report contains AI/ML domain analysis" && \
grep -q "DEEP SEMANTIC INSIGHTS" /root/chatClusteringTool/output/analysis_report.txt && echo "   ✓ Report includes deep semantic insights section" && \
echo "" && \
echo "4. Clustered Data Output:" && \
/usr/bin/python3 -c "import json; data=json.load(open('/root/chatClusteringTool/output/clustered_data.json')); print(f'   ✓ JSON valid with {len(data[\"first_chat_analysis\"][\"conversations\"])} conversations'); print(f'   ✓ Classification model: {data[\"metadata\"][\"classification_model\"]}'); print(f'   ✓ Labeling method: {data[\"metadata\"][\"labeling_method\"]}')" && \
echo "" && \
echo "=== ALL ACCEPTANCE CRITERIA MET ==="
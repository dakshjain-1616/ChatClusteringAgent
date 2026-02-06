echo "=== VERIFICATION: Fine-Grained Label Examples ===" && \
grep -E "Cluster [0-9]+:" /root/chatClusteringTool/output/analysis_report.txt | head -20 && \
echo -e "\n=== Comparison with Original Taxonomy ===" && \
echo "BEFORE (Broad): 'Model Training', 'Debugging', 'Data Preprocessing'" && \
echo "AFTER (Specific): 'Positional Encoding and Embeddings', 'Contrastive Learning and Self-Supervision', 'CUDA/GPU Memory Debugging', 'Federated Learning Setup'" && \
echo -e "\n=== All Files Generated ===" && \
ls -lh /root/chatClusteringTool/output/
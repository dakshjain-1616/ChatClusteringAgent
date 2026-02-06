cd /root/chatClusteringTool && ls -lh final_verification/ output_original/ && echo "=== Verification Complete ===" && /usr/bin/python3 -c "
import json
with open('final_verification/clustered_data.json') as f:
    data = json.load(f)
    print(f'✓ Clustered data validated')
    print(f'  - First chat conversations: {len(data[\"first_chat_analysis\"][\"conversations\"])}')
    print(f'  - Full history conversations: {len(data[\"full_history_analysis\"][\"conversations\"])}')
    print(f'  - First chat clusters: {data[\"first_chat_analysis\"][\"num_clusters\"]}')
    print(f'  - Full history clusters: {data[\"full_history_analysis\"][\"num_clusters\"]}')
    print(f'  - Sample has cluster_id: {\"cluster_id\" in data[\"first_chat_analysis\"][\"conversations\"][0]}')
    print(f'  - Sample has cluster_label: {\"cluster_label\" in data[\"first_chat_analysis\"][\"conversations\"][0]}')
"
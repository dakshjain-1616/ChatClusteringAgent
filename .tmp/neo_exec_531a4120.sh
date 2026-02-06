ls -lh /root/chatClusteringTool/
wc -l /root/chatClusteringTool/last_50_chats.json
/usr/bin/python3 -c "import json; data=json.load(open('/root/chatClusteringTool/last_50_chats.json')); print(f'Total conversations: {len(data)}'); print(f'Sample structure keys: {list(data[0].keys())}'); print(f'Message structure: {list(data[0][\"messages\"][0].keys())}')"
import json
import logging
from typing import Dict, List, Tuple
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChatDataLoader:
    def __init__(self, json_path: str):
        self.json_path = Path(json_path)
        self.raw_data = None
        
    def load_data(self) -> List[Dict]:
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                self.raw_data = json.load(f)
            logger.info(f"Loaded {len(self.raw_data)} conversations from {self.json_path}")
            return self.raw_data
        except FileNotFoundError:
            logger.error(f"File not found: {self.json_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format: {e}")
            raise
    
    def extract_user_messages(self, conversation: Dict) -> List[str]:
        messages = conversation.get('messages', [])
        user_messages = []
        
        for msg in messages:
            if msg.get('sender') == 'User':
                content = msg.get('content', '').strip()
                if content:
                    user_messages.append(content)
        
        return user_messages
    
    def parse_conversations(self) -> Tuple[List[Dict], List[Dict]]:
        if self.raw_data is None:
            self.load_data()
        
        first_chat_data = []
        full_history_data = []
        
        for idx, conversation in enumerate(self.raw_data):
            user_messages = self.extract_user_messages(conversation)
            
            if not user_messages:
                logger.warning(f"Conversation {idx} has no user messages, skipping")
                continue
            
            thread_id = conversation.get('thread_id', f'thread_{idx}')
            user_id = conversation.get('user_id', f'user_{idx}')
            
            first_chat_text = user_messages[0]
            first_chat_data.append({
                'id': thread_id,
                'user_id': user_id,
                'text': first_chat_text,
                'conversation_index': idx,
                'message_count': len(user_messages)
            })
            
            full_history_text = ' '.join(user_messages)
            full_history_data.append({
                'id': thread_id,
                'user_id': user_id,
                'text': full_history_text,
                'conversation_index': idx,
                'message_count': len(user_messages)
            })
        
        logger.info(f"Parsed {len(first_chat_data)} first chats and {len(full_history_data)} full histories")
        return first_chat_data, full_history_data
    
    def get_statistics(self) -> Dict:
        if self.raw_data is None:
            self.load_data()
        
        stats = {
            'total_conversations': len(self.raw_data),
            'conversations_with_messages': 0,
            'total_messages': 0,
            'total_user_messages': 0,
            'avg_messages_per_conversation': 0
        }
        
        message_counts = []
        for conv in self.raw_data:
            messages = conv.get('messages', [])
            if messages:
                stats['conversations_with_messages'] += 1
                stats['total_messages'] += len(messages)
                user_msg_count = sum(1 for m in messages if m.get('sender') == 'User')
                stats['total_user_messages'] += user_msg_count
                message_counts.append(user_msg_count)
        
        if message_counts:
            stats['avg_messages_per_conversation'] = sum(message_counts) / len(message_counts)
        
        return stats
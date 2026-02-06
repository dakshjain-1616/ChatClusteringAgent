import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def generate_large_dataset(input_file: str, output_file: str, target_size: int = 5000):
    logger.info(f"Loading base dataset from: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        base_data = json.load(f)
    
    base_size = len(base_data)
    logger.info(f"Base dataset size: {base_size}")
    
    if base_size == 0:
        logger.error("Base dataset is empty!")
        return
    
    logger.info(f"Generating synthetic dataset with {target_size} entries...")
    
    synthetic_data = []
    repetitions = (target_size // base_size) + 1
    
    for rep in range(repetitions):
        for idx, conversation in enumerate(base_data):
            if len(synthetic_data) >= target_size:
                break
            
            modified_conversation = conversation.copy()
            
            if 'conversation_id' in modified_conversation:
                modified_conversation['conversation_id'] = f"{modified_conversation['conversation_id']}_rep{rep}"
            
            synthetic_data.append(modified_conversation)
        
        if len(synthetic_data) >= target_size:
            break
    
    synthetic_data = synthetic_data[:target_size]
    
    logger.info(f"Generated {len(synthetic_data)} synthetic conversations")
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(synthetic_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Synthetic dataset saved to: {output_file}")
    logger.info(f"File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
    
    return output_file

if __name__ == '__main__':
    project_dir = Path(__file__).parent
    input_file = project_dir / 'last_50_chats.json'
    output_file = project_dir / 'large_dataset_5000.json'
    
    if not input_file.exists():
        logger.error(f"Base dataset not found: {input_file}")
        sys.exit(1)
    
    generate_large_dataset(str(input_file), str(output_file), target_size=5000)
    logger.info("="*80)
    logger.info("Dataset generation complete!")
    logger.info("="*80)
cd /root/chatClusteringTool && /usr/bin/python3 -c "
import os
import time
import subprocess

# Check if large dataset output exists
output_dir = '/root/chatClusteringTool/output_large'
expected_files = ['clustered_data.json', 'analysis_report.txt', 'first_chat_pie.png', 'first_chat_scatter.png', 'full_history_pie.png', 'full_history_scatter.png']

if os.path.exists(output_dir):
    existing_files = os.listdir(output_dir)
    print(f'Output directory exists with {len(existing_files)} files')
    
    if len(existing_files) >= 6:
        print('✓ All expected files present in output_large/')
        for f in expected_files:
            if f in existing_files:
                size = os.path.getsize(os.path.join(output_dir, f))
                print(f'  ✓ {f} ({size:,} bytes)')
    else:
        print(f'Partial output: {existing_files}')
        print('Large dataset may still be processing...')
else:
    print('Output directory does not exist yet - processing may still be running')

# Verify original dataset outputs
print('\\n=== Verifying Original Dataset Outputs ===')
original_dir = '/root/chatClusteringTool/output_original'
for f in expected_files:
    fpath = os.path.join(original_dir, f)
    if os.path.exists(fpath):
        size = os.path.getsize(fpath)
        print(f'✓ {f} ({size:,} bytes)')
"
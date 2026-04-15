import os
from pathlib import Path

TARGET_DIR = Path("/home/meow/Documents/Antigravity/Pollen_Detection_Template")

REPLACEMENTS = {
    "Colorado_pollen_detection": "Pollen_Detection_Template",
    "PEG/Colorado": "PROJECT_PREFIX"
}

def sanitize():
    for root, _, files in os.walk(TARGET_DIR):
        if ".git" in root or ".venv" in root or "docs" in root or "documentations" in root:
            continue
            
        for file in files:
            if not file.endswith(('.py', '.sh', '.yaml', '.yml', 'Dockerfile.species', 'Dockerfile.czi', 'Dockerfile.train')):
                continue
                
            file_path = Path(root) / file
            
            try:
                content = file_path.read_text(encoding='utf-8')
                new_content = content
                for old_val, new_val in REPLACEMENTS.items():
                    new_content = new_content.replace(old_val, new_val)
                    
                if new_content != content:
                    file_path.write_text(new_content, encoding='utf-8')
                    print(f"Sanitized: {file_path}")
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")

if __name__ == "__main__":
    sanitize()

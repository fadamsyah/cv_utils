import os
import shutil

from pathlib import Path

def boolean_string(s: str) -> bool:
    # Ngambil dari zyolo efficientdet
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def create_and_overwrite_dir(path_dir: str) -> None:
    # Create the directory
    Path(path_dir).mkdir(parents=True, exist_ok=True)
    
    # Overwrite the directory
    for path in os.listdir(path_dir):
        try: os.remove(os.path.join(path_dir, path))
        except IsADirectoryError: shutil.rmtree(os.path.join(path_dir, path))
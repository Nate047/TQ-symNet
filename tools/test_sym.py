"""Run the unchanged original Sym test using this project's local mmdet."""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == '__main__':
    runpy.run_path(str(ROOT / 'tools/test_sym_original.py'), run_name='__main__')

import os
import sys

# Automatically detect and add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_ROOT, 'datasets')

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
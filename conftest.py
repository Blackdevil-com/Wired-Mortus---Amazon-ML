# conftest.py — makes the project root importable during pytest
import sys
from pathlib import Path

# Insert project root so "from src.preprocessing.xxx import yyy" works
sys.path.insert(0, str(Path(__file__).parent))

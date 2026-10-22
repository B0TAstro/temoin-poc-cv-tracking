"""Put src/ on the path: the POC is run as scripts, not installed as a package."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

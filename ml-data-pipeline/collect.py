"""Entry point: collect data using DataCollectionAgent."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agents.data_collection_agent import DataCollectionAgent, main

if __name__ == "__main__":
    main()

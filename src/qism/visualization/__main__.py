"""Execution entrypoint for python -m qism.visualization."""

import sys
from scripts.generate_visualizations import main

if __name__ == "__main__":
    sys.exit(main())

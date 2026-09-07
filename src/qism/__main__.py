"""Main module execution entrypoint for python -m qism."""

import sys
from qism.cli import main

if __name__ == "__main__":
    sys.exit(main())

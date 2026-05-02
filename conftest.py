"""Make `dualcoeff` importable when running pytest from the repo root
without `pip install -e .`. Adds the parent directory of this package's
folder to sys.path so `import dualcoeff` resolves to the package whose
`__init__.py` lives alongside this file's directory."""
import os
import sys

PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

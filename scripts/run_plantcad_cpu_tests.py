#!/usr/bin/env python3
import sys,unittest,importlib.util
from pathlib import Path
root=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(root/'src'))
suite=unittest.defaultTestLoader.discover(str(root/'tests'), pattern='test_plantcad*.py')
result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)

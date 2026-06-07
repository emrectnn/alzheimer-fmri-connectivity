"""Convenience launcher for the analysis pipeline."""

import sys
import os

KOD_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, KOD_DIR)

from importlib import import_module

print("Dinamik ozne kesfi basliyor...")
discover = import_module("00_discover")
result   = discover.run(verbose=True)

if not result["all_valid"]:
    print("HATA: Hicbir gecerli ozne bulunamadi.")
    sys.exit(1)

main = import_module("main")
main.run_full_pipeline(convert_only=False)

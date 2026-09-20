#!/usr/bin/env python3
"""CLI for the CPU-only streaming PlantRegMap PhyloP lookup."""

import argparse
import importlib.util
from pathlib import Path

module_path = Path(__file__).resolve().parents[1] / "src" / "popgenlm" / "conservation.py"
spec = importlib.util.spec_from_file_location("popgenlm_conservation", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
lookup_bedgraph = module.lookup_bedgraph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", required=True)
    parser.add_argument("--track", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata", required=True)
    args = parser.parse_args()
    lookup_bedgraph(args.variants, args.track, args.output, args.metadata)


if __name__ == "__main__":
    main()

"""Production entry point; heavy dependencies are optional until preflight/run."""

from __future__ import annotations

import argparse
import json

from .plantcad_run import preflight, run


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline chromosome-wise PlantCAD scoring")
    parser.add_argument("--config", required=True, help="Absolute immutable run configuration JSON")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate imports, inputs and outputs; never load weights or use GPU",
    )
    args = parser.parse_args(argv)
    if args.preflight_only:
        print(json.dumps(preflight(args.config)[3], indent=2))
    else:
        run(args.config)


if __name__ == "__main__":
    main()

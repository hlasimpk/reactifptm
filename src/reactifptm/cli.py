"""Command-line interface for reactifPTM."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Sequence

from .core import Reactifptm

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reactifptm",
        description="Calculate reactifPTM scores from predicted model outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Example:
  reactifptm scores.json model.pdb -o results.json
""",
    )
    parser.add_argument(
        "input",
        help="PAE file (npz, npy, json, or pickle) containing the predicted aligned error matrix.",
    )
    parser.add_argument(
        "structure",
        help="Structure file (.pdb or .cif).",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=8.0,
        help="Cb-Cb contact threshold in Angstroms (default: 8.0).",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output JSON file path. If omitted, results are printed only.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

    try:
        reactifptm = Reactifptm(args.input, args.structure, threshold=args.threshold)
        overall, pairwise = reactifptm.compute_reactifptm()
    except ValueError as e:
        logger.error(e)
        return 1

    print(f"reactifpTM: {overall}")
    if overall == 0.0:
        logger.warning(
            "no inter-chain atom pairs found within %g Å. "
            "The chains may not be in contact at this threshold — try increasing it with -t.",
            args.threshold,
        )
    print("\npairwise reactifpTM (directional):")
    for key, value in pairwise.items():
        print(f"  {key}: {value}")

    print("\npairwise reactifpTM (max of both directions):")
    for key, value in reactifptm.reactifptm_pairwise_max.items():
        print(f"  {key}: {value}")

    if args.output:
        reactifptm.save_results(args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())

import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

import importlib.util
module_path = Path(__file__).resolve().parents[1] / "src" / "popgenlm" / "conservation.py"
module_spec = importlib.util.spec_from_file_location("popgenlm_conservation", module_path)
module = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(module)
TrackContractError = module.TrackContractError
lookup_bedgraph = module.lookup_bedgraph


class ConservationLookupTests(unittest.TestCase):
    def write_fixture(self, track, variants):
        root = Path(tempfile.mkdtemp())
        track_path = root / "track.bedGraph.gz"
        with gzip.open(track_path, "wt") as handle:
            handle.write(track)
        variant_path = root / "variants.tsv"
        with variant_path.open("w") as handle:
            handle.write("chrom\tpos\tref\talt\n")
            handle.write(variants)
        return root, track_path, variant_path

    def test_zero_based_half_open_boundary_and_negative_score(self):
        # Track rows [116,117), [117,118), [118,119); POS 117 maps to [116,117).
        root, track, variants = self.write_fixture(
            "Chr1 116 117 0.281094765673459\n"
            "Chr1 117 118 0.308706582709583\n"
            "Chr1 118 119 -0.928163664650044\n",
            "1\t116\tA\tG\n1\t117\tA\tG\n1\t118\tC\tT\n1\t119\tG\tA\n1\t120\tT\tC\n",
        )
        out, meta = root / "out.tsv", root / "meta.json"
        summary = lookup_bedgraph(variants, track, out, meta)
        with out.open() as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        self.assertEqual([r["phylop"] for r in rows], ["", "0.281094765673459", "0.308706582709583", "-0.928163664650044", ""])
        self.assertEqual([r["status"] for r in rows], ["missing_coverage", "covered", "covered", "covered", "missing_coverage"])
        self.assertEqual(summary["covered_rows"], 3)
        self.assertEqual(json.loads(meta.read_text())["missing_coverage_rows"], 2)

    def test_overlap_and_order_are_rejected(self):
        root, track, variants = self.write_fixture(
            "Chr1 10 20 1\nChr1 19 21 2\n", "1\t11\tA\tG\n")
        with self.assertRaises(TrackContractError):
            lookup_bedgraph(variants, track, root / "out.tsv")

    def test_unknown_chromosome_and_duplicate_keys_are_rejected(self):
        root, track, variants = self.write_fixture("Chr1 10 11 1\n", "1\t11\tA\tG\n1\t11\tA\tG\n")
        with self.assertRaises(ValueError):
            lookup_bedgraph(variants, track, root / "out.tsv")
        root, track, variants = self.write_fixture("Chr1 10 11 1\n", "6\t11\tA\tG\n")
        with self.assertRaises(ValueError):
            lookup_bedgraph(variants, track, root / "out.tsv")

    def test_track_bounds_are_rejected(self):
        root, track, variants = self.write_fixture("Chr1 30427671 30427672 1\n", "1\t1\tA\tG\n")
        with self.assertRaises(TrackContractError):
            lookup_bedgraph(variants, track, root / "out.tsv")


if __name__ == "__main__":
    unittest.main()

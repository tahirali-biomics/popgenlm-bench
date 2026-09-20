"""CPU-only controlled test data for production validation; no model loading."""

import tempfile
import unittest
from pathlib import Path

from popgenlm.plantcad import coordinate_to_context, population_scores, validate_reference
from popgenlm.plantcad_run import (
    check_resume,
    compare_overlap,
    digest,
    load_variants,
    validate_scores,
    write_json,
    write_tsv,
)


def variant(pos=2):
    return {
        "chrom": "1",
        "pos": str(pos),
        "ref": "C",
        "alt": "T",
        "fasta_chrom": "chr1",
        "af_alt": "0.1",
        "ac_alt": "2",
        "an": "20",
        "variant_id": f"1:{pos}:C:T",
    }


def scored(row):
    return {
        **row,
        **population_scores(row["ref"], row["alt"], row["af_alt"], -2.0),
        "ref_logprob": -1.0,
        "alt_logprob": -3.0,
    }


class ProductionTests(unittest.TestCase):
    def test_all_population_orientations_including_tie(self):
        for af, minor, orient in [
            (0.1, -2.0, "alt_minor"),
            (0.9, 2.0, "ref_minor"),
            (0.5, None, "frequency_tie"),
        ]:
            with self.subTest(af=af):
                p = population_scores("C", "T", af, -2.0)
                self.assertEqual(p["plantcad_score_ref_alt"], -2.0)
                self.assertEqual(p["plantcad_score_ref_alt_negative"], 2.0)
                self.assertEqual(p["plantcad_score_minor_vs_major"], minor)
                self.assertEqual(p["orientation"], orient)
                self.assertEqual(p["plantcad_score_paper_oriented"], 2.0 if af > 0.5 else -2.0)
                if af == 0.5:
                    self.assertIsNone(p["minor_allele"])
                    self.assertIsNone(p["major_allele"])

    def test_boundary_padding_and_ref_rejection(self):
        for pos, ref in [(1, "A"), (4, "T")]:
            c, idx = coordinate_to_context("ACGT", pos)
            self.assertEqual((len(c), idx, c[idx]), (512, 255, ref))
        with self.assertRaises(ValueError):
            validate_reference("ACGT", 1, "C")
        with self.assertRaises(ValueError):
            validate_reference("ACGT", 5, "A")

    def test_mapping_count_and_duplicate_input_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "rows.tsv"
            rows = [variant(3), variant(2)]
            write_tsv(path, rows)
            mapping = {"1": {"fasta_chrom": "chr1", "length": 4}}
            self.assertEqual([r["pos"] for r in load_variants(path, mapping, 2)], ["2", "3"])
            with self.assertRaises(ValueError):
                load_variants(path, mapping, 1)
            with self.assertRaises(ValueError):
                load_variants(path, {"1": {"fasta_chrom": "wrong", "length": 4}}, 2)
            duplicate = Path(d) / "duplicate.tsv"
            write_tsv(duplicate, [variant(), variant()])
            with self.assertRaises(ValueError):
                load_variants(duplicate, mapping, 2)

    def test_missing_duplicate_and_unexpected_scores(self):
        row = variant()
        for wrong in ([], [scored(row), scored(row)], [scored(variant(3))]):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                validate_scores(wrong, [row])

    def test_nonfinite_and_wrong_orientation_rejected(self):
        for field, value in [
            ("plantcad_score_ref_alt", float("nan")),
            ("plantcad_score_ref_alt_negative", -2.0),
            ("plantcad_score_minor_vs_major", float("inf")),
            ("alt_logprob", float("nan")),
        ]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                row = scored(variant())
                row[field] = value
                validate_scores([row], [variant()])

    def test_original_metadata_preserved(self):
        row = variant()
        row["selection_hash"] = "unchanged"
        scores = scored(row)
        validate_scores([scores], [row])
        scores["selection_hash"] = "different"
        with self.assertRaises(ValueError):
            validate_scores([scores], [row])

    def test_restart_requires_matching_fingerprint_and_output_hash(self):
        with tempfile.TemporaryDirectory() as d:
            table, marker = Path(d) / "chr.tsv", Path(d) / "done.json"
            self.assertIsNone(check_resume(table, marker, "cfg", [variant()]))
            write_tsv(table, [scored(variant())])
            with self.assertRaises(ValueError):
                check_resume(table, marker, "cfg", [variant()])
            write_json(marker, {"fingerprint": "cfg", "rows": 1, "sha256": digest(table)})
            self.assertEqual(len(check_resume(table, marker, "cfg", [variant()])), 1)
            with self.assertRaises(ValueError):
                check_resume(table, marker, "changed", [variant()])
            with table.open("a") as f:
                f.write("\n")
            with self.assertRaises(ValueError):
                check_resume(table, marker, "cfg", [variant()])

    def test_overlap_reports_intentional_tolerance_failure(self):
        # Controlled mock scores only; production always computes actual checkpoint scores.
        with tempfile.TemporaryDirectory() as d:
            archive = Path(d) / "archive.tsv"
            rows = [
                {
                    "variant_id": str(i),
                    "adapter_batch1": -2.0,
                    "adapter_batch8": -2.0,
                    "authors_reference": -2.0,
                }
                for i in range(100)
            ]
            write_tsv(archive, rows)
            results = [{"variant_id": str(i), "plantcad_score_ref_alt": -2.0} for i in range(100)]
            results[4]["plantcad_score_ref_alt"] = -2.1
            _, summary = compare_overlap(results, archive, 1e-4, 1e-4)
            self.assertEqual(summary["adapter_batch8"]["failed_variant_ids"], ["4"])
            self.assertAlmostEqual(summary["adapter_batch8"]["max_abs_difference"], 0.1)
            with self.assertRaises(ValueError):
                compare_overlap(results[:-1], archive, 1e-4, 1e-4)

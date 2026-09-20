import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).parents[2]


def load(name):
    path = ROOT / "scripts" / "v0.3" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProductionContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h3 = load("build_master_table")
        cls.h4 = load("annotate_genomic_context")
        cls.h5b = load("compute_ld_structure")
        cls.h5c = load("compute_model_concordance")
        cls.h6a = load("build_genotype_umap")
        cls.h6b = load("build_model_evidence_figure")

    def test_h3_normalized_exact_key_and_orientation(self):
        row = {"chrom": "1", "fasta_chrom": "", "pos": "7", "ref": "a", "alt": "c"}
        self.assertEqual(self.h3.normalized_key(row), ("NC_003070.9", 7, "A", "C"))
        self.assertEqual(self.h3.parse_bool(" TRUE "), True)
        self.assertEqual(self.h3.parse_bool("0"), False)

    def test_h4_attribute_decoding_and_precedence_inputs(self):
        attrs = self.h4.parse_attributes("ID=gene%3A1;gene=At1;gene_biotype=protein_coding")
        self.assertEqual(attrs["ID"], "gene:1")
        self.assertEqual(
            self.h4.variant_key({"fasta_chrom": "NC_003070.9", "pos": "2", "ref": "A", "alt": "G"}),
            ("NC_003070.9", 2, "A", "G"),
        )

    def test_h5b_pairwise_complete_and_nonvariable(self):
        x = __import__("numpy").array([0, 1, -1, 2], dtype="int8")
        y = __import__("numpy").array([0, 1, 1, 2], dtype="int8")
        n, r2, status = self.h5b.pairwise_r2(x, y, minimum_complete=2)
        self.assertEqual((n, status), (3, "ok"))
        self.assertAlmostEqual(r2, 1.0)
        n, r2, status = self.h5b.pairwise_r2(
            __import__("numpy").array([1, 1], dtype="int8"),
            __import__("numpy").array([0, 2], dtype="int8"),
            minimum_complete=2,
        )
        self.assertEqual((n, r2, status), (2, None, "pairwise_nonvariable"))

    def test_h5c_descriptive_contracts_and_substitution(self):
        import numpy as np

        mask = np.ones(4, dtype=bool)
        n, estimate = self.h5c.correlation(np.arange(4.0), np.arange(4.0), mask, "pearson")
        self.assertEqual(n, 4)
        self.assertAlmostEqual(estimate, 1.0)
        self.assertEqual(self.h5c.substitution_class({"ref": "A", "alt": "C"}), "T>G")
        self.assertFalse("p_value" in self.h5c.COMPARISONS)

    def test_h6a_preprocessing_and_fixed_settings(self):
        import numpy as np

        result = self.h6a.preprocess(
            np.array([[0.0, 1.0, 2.0, 1.0], [2.0, 0.0, 1.0, 2.0], [0.0, 2.0, 1.0, 0.0]])
        )
        self.assertEqual(result.shape, (3, 4))
        self.assertEqual(self.h6a.SEED, 20260920)
        self.assertEqual(self.h6a.PCA_COMPONENTS, 50)
        self.assertEqual(self.h6a.UMAP_NEIGHBORS, 30)
        self.assertEqual(self.h6a.UMAP_MIN_DIST, 0.10)

    def test_h6b_figure_record_selection(self):
        rows = [{"scope": "unpruned", "comparison": "x", "method": "pearson", "estimate": "0.5"}]
        self.assertEqual(self.h6b.record(rows, scope="unpruned")["estimate"], "0.5")


if __name__ == "__main__":
    unittest.main()

"""CPU regression tests: small local tokenizer definition, no weights or downloads."""

import importlib.util
import json
import unittest
from pathlib import Path

from popgenlm.plantcad import coordinate_to_context, prepare_masked_inputs


@unittest.skipUnless(
    all(importlib.util.find_spec(name) for name in ("torch", "tokenizers", "transformers")),
    "PyTorch tokenizer stack unavailable",
)
class TestIupacPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from tokenizers import Tokenizer
        from transformers import PreTrainedTokenizerFast

        cls.torch = torch
        fixtures = Path(__file__).parent / "fixtures"
        cls.make_tokenizer = staticmethod(
            lambda: PreTrainedTokenizerFast(
                tokenizer_object=Tokenizer.from_str(
                    (fixtures / "plantcad_tokenizer.json").read_text()
                ),
                unk_token="[UNK]",
                mask_token="[MASK]",
                pad_token="[PAD]",
            )
        )
        cls.rows = json.loads((fixtures / "plantcad_iupac_contexts.json").read_text())

    def test_observed_failure_contexts(self):
        tokenizer = self.make_tokenizer()
        for r in self.rows:
            with self.subTest(variant=r["variant_id"]):
                x = prepare_masked_inputs(tokenizer, [r["context"]], [r])
                self.assertEqual(tuple(x.shape), (1, 512))
                self.assertEqual(int(x[0, 255]), 1)
                for item in r["non_acgt"]:
                    self.assertEqual(int(x[0, item["index"]]), 2)

    def test_batch1_batch8_and_final_short(self):
        tok = self.make_tokenizer()
        rows = (self.rows * 5)[:9]
        alone = [prepare_masked_inputs(tok, [r["context"]], [r])[0] for r in rows]
        batches = []
        for i in range(0, len(rows), 8):
            rs = rows[i : i + 8]
            batches.extend(prepare_masked_inputs(tok, [r["context"] for r in rs], rs))
        self.assertEqual(len(batches), 9)
        self.assertTrue(all(self.torch.equal(a, b) for a, b in zip(alone, batches)))

    def test_boundary_padding(self):
        tok = self.make_tokenizer()
        for pos, ref in ((1, "A"), (4, "T")):
            ctx, _ = coordinate_to_context("ACGT", pos)
            masked = prepare_masked_inputs(tok, [ctx], [{"ref": ref}])
            self.assertEqual(int((masked == 2).sum()), 508)
            self.assertEqual(int((masked == 1).sum()), 1)

    def test_all_iupac_symbols_retain_positions(self):
        tok = self.make_tokenizer()
        ctx = list(("ACGTRYSWKMBDHVN" * 37)[:512])
        ctx[255] = "T"
        ctx = "".join(ctx)
        raw = tok([ctx], add_special_tokens=False, return_tensors="pt")["input_ids"]
        masked = prepare_masked_inputs(tok, [ctx], [{"ref": "T"}])
        self.assertEqual((raw != masked).nonzero().tolist(), [[0, 255]])
        for i, b in enumerate(ctx):
            if b not in "ACGT":
                self.assertEqual(int(masked[0, i]), 2)

    def test_case_normalization_before_preprocessing(self):
        tok = self.make_tokenizer()
        upper, _ = coordinate_to_context("ACGT" * 128, 256)
        lower, _ = coordinate_to_context("acgt" * 128, 256)
        self.assertEqual(upper, lower)
        self.assertTrue(
            self.torch.equal(
                prepare_masked_inputs(tok, [upper], [{"ref": "T"}]),
                prepare_masked_inputs(tok, [lower], [{"ref": "T"}]),
            )
        )

    def test_reject_shifted_and_truncated_tokens(self):
        tok = self.make_tokenizer()

        class Corrupt:
            def __init__(self, mode):
                self.mode = mode
                self.mask_token_id = tok.mask_token_id

            def get_vocab(self):
                return tok.get_vocab()

            def __call__(self, *a, **kw):
                result = tok(*a, **kw)
                x = result["input_ids"]
                result["input_ids"] = x.roll(1, dims=1) if self.mode == "shift" else x[:, :-1]
                return result

        for mode in ("shift", "drop"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                prepare_masked_inputs(Corrupt(mode), ["ACGT" * 128], [{"ref": "T"}])

    def test_reject_wrong_ref_and_non_dna_text(self):
        tok = self.make_tokenizer()
        with self.assertRaisesRegex(ValueError, "REF/target"):
            prepare_masked_inputs(tok, ["ACGT" * 128], [{"ref": "A"}])
        for bad in ("U", "X", "-", " ", "[MASK]"):
            ctx = ("A" * 10 + bad + "A" * 512)[:512]
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                prepare_masked_inputs(tok, [ctx], [{"ref": "A"}])

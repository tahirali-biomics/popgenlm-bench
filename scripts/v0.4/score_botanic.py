"""Pinned offline Botanic1-S LLRs; restartable chromosome/context shards."""

import argparse
import hashlib
import json
import os
import sys
import time
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
import pandas as pd
import torch
import transformers.utils.import_utils as optional_imports

# Genomic inference uses no vision. The inherited optional torchvision binary
# is incompatible; disable only its discovery in this isolated process.
optional_imports._torchvision_available = False
from transformers import AutoModelForMaskedLM, AutoTokenizer  # noqa: E402 -- preserve audited production execution/order

from popgenlm.regulatory import ALIASES, context, reverse_complement  # noqa: E402 -- preserve audited production execution/order


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--reference", default="/projects/ag-demeaux/tali/PopGenLM/data/reference/TAIR10.1_genomic.fna")
    p.add_argument("--chrom")
    p.add_argument("--batch", type=int, default=2)
    a = p.parse_args()
    run = Path(a.run)
    out = run / "results/botanic1_s"
    out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((run / "inputs/model_protocol.json").read_text())
    for name, m in json.loads((run / "inputs/frozen/freeze.json").read_text())["files"].items():
        assert sha(run / "inputs/frozen" / name) == m["sha256"]
    x = pd.read_csv(
        run / ("inputs/smoke_candidates.tsv" if a.smoke else "inputs/frozen/candidates.tsv"),
        sep="\t",
    )
    if a.chrom:
        x = x[x.chrom.astype(str) == a.chrom]
    seq = {}
    key = None
    with Path(a.reference).open() as fasta:
        for line in fasta:
            if line.startswith(">"):
                key = line[1:].split()[0]
                seq[key] = []
            else:
                seq[key].append(line.strip().upper())
    seq = {k: "".join(v) for k, v in seq.items()}
    assert torch.cuda.is_available(), "GPU required"
    torch.manual_seed(20261114)
    torch.backends.cuda.matmul.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(a.checkpoint, trust_remote_code=True, local_files_only=True)
    model = (
        AutoModelForMaskedLM.from_pretrained(
            a.checkpoint, trust_remote_code=True, local_files_only=True
        )
        .eval()
        .to("cuda")
    )
    assert (
        tok.prepend_cls_token
        and tok.cls_token_id is not None
        and tok.mask_token_id == model.config.mask_idx
    )
    bases = {b: tok.convert_tokens_to_ids(b) for b in "ACGT"}

    def score(strings, centers, refs, alts):
        encoded = tok(strings, padding=False, truncation=False, return_tensors="pt")[
            "input_ids"
        ].to("cuda")
        assert encoded.shape[1] == len(strings[0]) + 1 <= model.config.max_seqlen
        assert (encoded[:, 0] == tok.cls_token_id).all()
        masked = encoded.clone()
        for i, (c, r) in enumerate(zip(centers, refs)):
            assert strings[i][c] == r and int(encoded[i, c + 1]) == bases[r]
            masked[i, c + 1] = tok.mask_token_id
        assert int((masked != encoded).sum()) == len(strings)
        with torch.inference_mode():
            logits = model(input_ids=masked).logits.float()
        vals = []
        for i, (c, r, t) in enumerate(zip(centers, refs, alts)):
            v = logits[i, c + 1]
            assert torch.isfinite(v).all()
            llr = v[bases[t]] - v[bases[r]]
            lp = torch.log_softmax(v, dim=-1)
            assert torch.allclose(llr, lp[bases[t]] - lp[bases[r]], atol=1e-5)
            vals.append(float(llr))
        return vals

    start = time.time()
    runs = []
    for length in protocol["contexts_bp"]:
        for chrom, g in x.groupby("chrom", sort=True):
            target = out / (f"smoke_{length}.tsv" if a.smoke else f"chr{chrom}_{length}.tsv")
            done = target.with_suffix(".complete.json")
            if done.exists():
                assert sha(target) == json.loads(done.read_text())["sha256"]
                continue
            rows = []
            for i in range(0, len(g), a.batch):
                strings = []
                centers = []
                refs = []
                alts = []
                batch = g.iloc[i : i + a.batch]
                for r in batch.itertuples():
                    s, c = context(seq[ALIASES[str(r.chrom)]], r.pos, r.ref, length)
                    strings += [s, reverse_complement(s)]
                    centers += [c, length - 1 - c]
                    refs += [r.ref, reverse_complement(r.ref)]
                    alts += [r.alt, reverse_complement(r.alt)]
                values = score(strings, centers, refs, alts)
                if a.smoke:
                    single = [
                        score([s], [c], [r], [t])[0]
                        for s, c, r, t in zip(strings, centers, refs, alts)
                    ]
                    assert np.allclose(values, single, atol=1e-4, rtol=1e-4), "batch parity"
                    with torch.inference_mode():
                        official = model.score_variant_zero_shot(
                            sequence=strings[0],
                            tokenizer=tok,
                            variant_pos=centers[0],
                            variant_char=alts[0],
                        )
                    assert np.isclose(values[0], official, atol=1e-4, rtol=1e-4), (
                        "official helper parity"
                    )
                    # Two-strand aggregation is RC invariant even when single strands differ.
                    reverse = score(strings[::-1], centers[::-1], refs[::-1], alts[::-1])
                    assert np.allclose(values, reverse[::-1], atol=1e-4, rtol=1e-4), (
                        "RC order parity"
                    )
                for j, r in enumerate(batch.itertuples()):
                    fw, rc = values[2 * j : 2 * j + 2]
                    raw = (fw + rc) / 2
                    rows.append(
                        {
                            "variant_id": r.variant_id,
                            "context_bp": length,
                            "llr_forward": fw,
                            "llr_rc": rc,
                            "raw_alt_minus_ref": raw,
                            "magnitude": abs(raw),
                            "minor_minus_major": -raw if r.af_alt > 0.5 else raw,
                        }
                    )
                if i % 100 == 0:
                    print(
                        "PROGRESS",
                        chrom,
                        length,
                        i,
                        len(g),
                        "seconds",
                        round(time.time() - start, 2),
                        flush=True,
                    )
            pd.DataFrame(rows).to_csv(target, sep="\t", index=False)
            m = {
                "sha256": sha(target),
                "rows": len(rows),
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "elapsed": time.time() - start,
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                "smoke": a.smoke,
                "context_bp": length,
            }
            done.write_text(json.dumps(m, indent=2))
            runs.append(m)
    env = {
        "checkpoint": a.checkpoint,
        "revision": protocol["botanic_revision"],
        "python": sys.executable,
        "versions": {n: metadata.version(n) for n in ["torch", "transformers", "numpy", "pandas"]},
        "gpu": torch.cuda.get_device_name(),
        "pure_torch": os.environ.get("BOTANIC1_FORCE_PURE_TORCH"),
        "CLS_offset": 1,
        "padding": False,
        "orientation": protocol["llr"],
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "runs": runs,
    }
    (out / ("smoke_manifest.json" if a.smoke else f"manifest_chr{a.chrom}.json")).write_text(
        json.dumps(env, indent=2)
    )
    print("PASS Botanic", a.smoke, flush=True)


if __name__ == "__main__":
    main()

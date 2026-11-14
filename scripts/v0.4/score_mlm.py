"""Offline, restartable masked allele LLR scoring, with independent smoke mode."""

import argparse
import hashlib
import importlib
import json
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
import pandas as pd
import torch
from popgenlm.regulatory import context, reverse_complement
from transformers import AutoModelForMaskedLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--model", required=True)
p.add_argument("--run", required=True)
p.add_argument("--smoke", action="store_true")
p.add_argument("--lengths", default="512,4096")
p.add_argument("--chrom")
p.add_argument("--batch-variants", type=int, default=2)
a = p.parse_args()
run = Path(a.run)
out = run / "results" / a.model
out.mkdir(parents=True, exist_ok=True)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(c)
    return h.hexdigest()


lengths = list(map(int, a.lengths.split(",")))
if not a.smoke:
    frozen = run / "inputs/frozen"
    freeze = json.loads((frozen / "freeze.json").read_text())
    for n, v in freeze["files"].items():
        assert digest(frozen / n) == v["sha256"], n
    candidates = pd.read_csv(frozen / "candidates.tsv", sep="\t")
    if a.chrom:
        candidates = candidates[candidates.chrom.astype(str) == a.chrom]
    candidates = candidates.drop_duplicates("variant_id")
    seq = {}
    name = None
    root = Path("/projects/ag-demeaux/tali/PopGenLM")
    with open(root / "data/reference/TAIR10.1_genomic.fna") as f:
        for line in f:
            if line.startswith(">"):
                name = line[1:].split()[0]
                seq[name] = []
            else:
                seq[name].append(line.strip().upper())
    seq = {k: "".join(x) for k, x in seq.items()}
    from popgenlm.regulatory import ALIASES
else:
    candidates = pd.DataFrame(
        [
            {
                "variant_id": "technical:1:A:C",
                "chrom": "fixture",
                "pos": 5001,
                "ref": "A",
                "alt": "C",
                "af_alt": 0.25,
            },
            {
                "variant_id": "technical:2:C:T",
                "chrom": "fixture",
                "pos": 5002,
                "ref": "C",
                "alt": "T",
                "af_alt": 0.75,
            },
        ]
    )
    seq = {"fixture": "ACGT" * 3000}
    ALIASES = {"fixture": "fixture"}
if a.model == "gpn_brassicales":
    sys.path.insert(0, "/projects/ag-demeaux/tali/gpn/songlab-cal_gpn")
    importlib.import_module("gpn.model")  # register the pinned ConvNet architecture
assert torch.cuda.is_available(), "GPU required for this scoring run"
torch.manual_seed(20261114)
tokenizer = AutoTokenizer.from_pretrained(
    a.checkpoint, trust_remote_code=True, local_files_only=True
)
model = (
    AutoModelForMaskedLM.from_pretrained(
        a.checkpoint, trust_remote_code=True, local_files_only=True, torch_dtype=torch.float32
    )
    .eval()
    .to("cuda")
)
assert tokenizer.mask_token_id is not None
base_ids = {b: tokenizer.encode(b, add_special_tokens=False)[0] for b in "ACGT"}
assert len(set(base_ids.values())) == 4
print(
    "MODEL_LOADED", a.model, "device", torch.cuda.get_device_name(0), "bases", base_ids, flush=True
)


def score_batch(strings, centers, refs, alts):
    encoded = tokenizer(strings, return_tensors="pt", padding=False, add_special_tokens=False)[
        "input_ids"
    ].to("cuda")
    assert encoded.shape[1] == len(strings[0]), "one nucleotide must equal one token"
    for i, (ref, center) in enumerate(zip(refs, centers)):
        assert int(encoded[i, center]) == base_ids[ref]
    masked = encoded.clone()
    for i, center in enumerate(centers):
        masked[i, center] = tokenizer.mask_token_id
    assert int((masked != encoded).sum()) == len(strings)
    with torch.inference_mode():
        logits = model(input_ids=masked).logits.float()
    result = []
    for i, (center, ref, alt) in enumerate(zip(centers, refs, alts)):
        x = logits[i, center]
        assert torch.isfinite(x).all()
        llr = x[base_ids[alt]] - x[base_ids[ref]]
        lp = torch.log_softmax(x, dim=-1)
        assert torch.allclose(llr, lp[base_ids[alt]] - lp[base_ids[ref]], atol=1e-5)
        result.append(float(llr))
    return result


start = time.time()
metadata = []
for length in lengths:
    for chrom, group in candidates.groupby("chrom", sort=True):
        dest = out / f"chr{chrom}_{length}.tsv"
        done = out / f"chr{chrom}_{length}.complete.json"
        if not a.smoke and done.exists():
            saved = json.loads(done.read_text())
            assert digest(dest) == saved["sha256"]
            print("REUSE", dest, flush=True)
            continue
        result = []
        for i in range(0, len(group), a.batch_variants):
            batch = group.iloc[i : i + a.batch_variants]
            strings = []
            centers = []
            refs = []
            alts = []
            for row in batch.itertuples():
                s, center = context(seq[ALIASES[str(row.chrom)]], row.pos, row.ref, length)
                if a.model == "gpn_brassicales":
                    center = length // 2
                    startpos = row.pos - 1 - center
                    s = seq[ALIASES[str(row.chrom)]][startpos : startpos + length]
                    assert s[center] == row.ref
                strings.extend([s, reverse_complement(s)])
                centers.extend([center, length - 1 - center])
                refs.extend([row.ref, reverse_complement(row.ref)])
                alts.extend([row.alt, reverse_complement(row.alt)])
            values = score_batch(strings, centers, refs, alts)
            if a.smoke:
                singles = [
                    score_batch([s], [c], [r], [alt])[0]
                    for s, c, r, alt in zip(strings, centers, refs, alts)
                ]
                assert np.allclose(values, singles, atol=1e-4, rtol=1e-4), "batch parity"
                if a.model != "gpn_brassicales":
                    assert np.max(np.abs(np.array(values)[::2] - np.array(values)[1::2])) < 1e-3, (
                        "RC-equivariance"
                    )
            for j, row in enumerate(batch.itertuples()):
                fw, rc = values[2 * j : 2 * j + 2]
                raw = (fw + rc) / 2
                result.append(
                    {
                        "variant_id": row.variant_id,
                        "model": a.model,
                        "context_bp": length,
                        "llr_forward": fw,
                        "llr_rc": rc,
                        "raw_alt_minus_ref": raw,
                        "minor_minus_major": -raw if row.af_alt > 0.5 else raw,
                        "magnitude": abs(raw),
                    }
                )
            if i % 100 == 0:
                print("PROGRESS", chrom, length, i, len(group), flush=True)
        table = pd.DataFrame(result)
        assert np.isfinite(table.raw_alt_minus_ref).all()
        assert table.variant_id.is_unique
        target = out / f"smoke_{length}.tsv" if a.smoke else dest
        table.to_csv(target, sep="\t", index=False)
        summary = {
            "sha256": digest(target),
            "rows": len(table),
            "elapsed_seconds": time.time() - start,
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            "checkpoint": a.checkpoint,
            "context_bp": length,
            "model": a.model,
            "smoke": a.smoke,
        }
        if not a.smoke:
            done.write_text(json.dumps(summary, indent=2) + "\n")
        metadata.append(summary)
versions = {}
import importlib.metadata as m

for n in ["torch", "transformers", "numpy", "pandas", "mamba-ssm", "causal-conv1d"]:
    try:
        versions[n] = m.version(n)
    except m.PackageNotFoundError:
        pass
manifest = {
    "model": a.model,
    "checkpoint": a.checkpoint,
    "weights": {p.name: digest(p) for p in Path(a.checkpoint).iterdir() if p.is_file()},
    "versions": versions,
    "python": sys.executable,
    "platform": platform.platform(),
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "runs": metadata,
    "orientation": "masked ALT-minus-REF logits; forward and RC averaged; magnitude for ranking",
}
(
    out
    / (
        "smoke_manifest.json"
        if a.smoke
        else ("manifest_chr" + a.chrom + ".json" if a.chrom else "manifest.json")
    )
).write_text(json.dumps(manifest, indent=2) + "\n")
print(
    "PASS",
    a.model,
    "smoke" if a.smoke else "production",
    "elapsed",
    time.time() - start,
    flush=True,
)

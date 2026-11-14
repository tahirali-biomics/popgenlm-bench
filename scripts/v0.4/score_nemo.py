"""Pinned released B. napus nemo transfer; paired CDS-masked REF/ALT inputs."""

import argparse
import ast
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--bundle", required=True)
p.add_argument("--smoke", action="store_true")
a = p.parse_args()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from popgenlm.regulatory import ALIASES, reverse_complement

run = Path(a.run)
out = run / "results/nemo_bnapus_transfer"
out.mkdir(exist_ok=True, parents=True)
bundle = Path(a.bundle)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# Execute only the audited pure architecture definition, never the upstream CLI.
s = (bundle / "nemo.py").read_text()
node = next(
    n for n in ast.parse(s).body if isinstance(n, ast.FunctionDef) and n.name == "build_nemo"
)
code = ast.get_source_segment(s, node)
namespace = {
    name: getattr(tf.keras.layers, name)
    for name in [
        "Input",
        "Conv1D",
        "AveragePooling1D",
        "Flatten",
        "Concatenate",
        "Dense",
        "BatchNormalization",
        "Dropout",
        "Activation",
    ]
}
namespace.update(Model=tf.keras.Model, ceil=math.ceil)
exec(compile(code, str(bundle / "nemo.py") + ":build_nemo", "exec"), namespace)  # noqa: S102 -- pinned audited architecture only
model = namespace["build_nemo"]()
model.load_weights(str(bundle / "nemo_v1.0_weights.h5"))


def encode(strings):
    arr = np.zeros((len(strings), 6200, 4), np.float32)
    for i, s in enumerate(strings):
        assert len(s) == 6200
        for j, b in enumerate(s):
            if b in "ACGT":
                arr[i, j, "ACGT".index(b)] = 1
    return arr


if a.smoke:
    ref = "ACGT" * 1550
    alt = ref[:5000] + "C" + ref[5001:]
    x = encode([ref, alt, ref])
    z = np.array(model([x, x], training=False)).reshape(-1)
    assert np.isfinite(z).all() and np.isclose(z[0], z[2], atol=1e-6)
    single = float(np.array(model([x[:1], x[:1]], training=False)).reshape(-1)[0])
    assert np.isclose(single, z[0], atol=1e-5)
    (out / "smoke_manifest.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "predictions": z.tolist(),
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "tensorflow": tf.__version__,
                "weights_sha256": sha(bundle / "nemo_v1.0_weights.h5"),
            },
            indent=2,
        )
        + "\n"
    )
    print("PASS nemo smoke")
    sys.exit(0)
frozen = run / "inputs/frozen"
freeze = json.loads((frozen / "freeze.json").read_text())
for n, v in freeze["files"].items():
    assert sha(frozen / n) == v["sha256"]
candidates = pd.read_csv(frozen / "candidates.tsv", sep="\t")
loci = pd.read_csv(frozen / "loci.tsv", sep="\t").set_index("gene")
root = Path("/projects/ag-demeaux/tali/PopGenLM")
seq = {}
name = None
with open(root / "data/reference/TAIR10.1_genomic.fna") as f:
    for line in f:
        if line.startswith(">"):
            name = line[1:].split()[0]
            seq[name] = []
        else:
            seq[name].append(line.strip().upper())
seq = {k: "".join(v) for k, v in seq.items()}
# Mask all annotated nuclear CDS, matching the published prediction bundle.
import gzip
import re

mask = {k: np.zeros(len(v), bool) for k, v in seq.items()}
transcripts = {}
with gzip.open(
    root / "data/incoming/GCF_000001735.4_TAIR10.1/GCF_000001735.4_TAIR10.1_genomic.gff.gz", "rt"
) as f:
    for line in f:
        if line.startswith("#"):
            continue
        fields = line.rstrip().split("\t")
        if len(fields) != 9 or fields[0] not in ALIASES.values():
            continue
        chrom = fields[0]
        kind = fields[2]
        st, en = int(fields[3]), int(fields[4])
        if kind == "CDS":
            mask[chrom][st - 1 : en] = True
        if kind == "mRNA":
            parent = re.search(r"Parent=gene-(AT[1-5]G\d{5})", fields[8])
            ident = re.search(r"(?:^|;)ID=([^;]+)", fields[8])
            if parent and parent[1] in loci.index:
                gene = parent[1]
                transcripts.setdefault(gene, []).append((en - st + 1, ident[1], st, en, fields[6]))
results = []
start = time.time()
for gene, g in candidates.groupby("gene", sort=True):
    dest = out / (gene + ".tsv")
    done = out / (gene + ".complete.json")
    if done.exists():
        assert sha(dest) == json.loads(done.read_text())["sha256"]
        continue
    if gene not in transcripts:
        for row in g.itertuples():
            results.append(
                {
                    "gene": gene,
                    "variant_id": row.variant_id,
                    "status": "no_mRNA",
                    "delta_log10_tpm": np.nan,
                }
            )
        continue
    _, transcript, st, en, strand = min(transcripts[gene], key=lambda t: (-t[0], t[1]))
    chrom = ALIASES[str(int(loci.loc[gene, "chrom"]))]
    sequence = seq[chrom]
    intervals = (
        [(st - 1 - 5000, st - 1 + 1200), (en - 1200, en + 5000)]
        if strand == "+"
        else [(en - 1200, en + 5000), (st - 1 - 5000, st - 1 + 1200)]
    )
    strings = []
    for lo, hi in intervals:
        arr = list(sequence[max(0, lo) : min(len(sequence), hi)])
        for j in np.where(mask[chrom][max(0, lo) : min(len(sequence), hi)])[0]:
            arr[j] = "N"
        s = "N" * max(0, -lo) + "".join(arr) + "N" * max(0, hi - len(sequence))
        assert len(s) == 6200
        strings.append(reverse_complement(s) if strand == "-" else s)
    refpred = float(
        np.array(model([encode([strings[0]]), encode([strings[1]])], training=False)).reshape(-1)[0]
    )
    eligible = []
    altered = []
    local = []
    for row in g.itertuples():
        position = row.pos - 1
        covered = any(lo <= position < hi for lo, hi in intervals)
        status = (
            "covered"
            if covered and not mask[chrom][position]
            else "CDS_masked"
            if covered
            else "outside_windows"
        )
        record = {
            "gene": gene,
            "variant_id": row.variant_id,
            "transcript": transcript,
            "status": status,
            "ref_scaled": refpred,
            "alt_scaled": np.nan,
            "delta_log10_tpm": np.nan,
        }
        if status == "covered":
            pair = strings.copy()
            for j, (lo, hi) in enumerate(intervals):
                if lo <= position < hi:
                    index = position - lo if strand == "+" else 6199 - (position - lo)
                    ref = row.ref if strand == "+" else reverse_complement(row.ref)
                    alt = row.alt if strand == "+" else reverse_complement(row.alt)
                    assert pair[j][index] == ref
                    pair[j] = pair[j][:index] + alt + pair[j][index + 1 :]
            altered.append(pair)
            eligible.append(len(local))
        local.append(record)
    for i in range(0, len(altered), 16):
        batch = altered[i : i + 16]
        values = np.array(
            model([encode([x[0] for x in batch]), encode([x[1] for x in batch])], training=False)
        ).reshape(-1)
        assert np.isfinite(values).all()
        for j, z in enumerate(values):
            record = local[eligible[i + j]]
            record["alt_scaled"] = float(z)
            record["delta_log10_tpm"] = float((z - refpred) * 1.0498368135592189)
    table = pd.DataFrame(local)
    table.to_csv(dest, sep="\t", index=False)
    done.write_text(
        json.dumps(
            {"sha256": sha(dest), "rows": len(table), "job_id": os.environ.get("SLURM_JOB_ID")},
            indent=2,
        )
        + "\n"
    )
    print("GENE", gene, len(eligible), "/", len(g), flush=True)
if results:
    pd.DataFrame(results).to_csv(out / "uncovered.tsv", sep="\t", index=False)
(out / "manifest.json").write_text(
    json.dumps(
        {
            "model": "nemo Bnapus v1.0 transfer",
            "source_revision": "c8deaaf18a8bd174300bf40e11fa2c69c0d6bb4e",
            "weights_sha256": sha(bundle / "nemo_v1.0_weights.h5"),
            "architecture_sha256": hashlib.sha256(code.encode()).hexdigest(),
            "tensorflow": tf.__version__,
            "python": sys.executable,
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "elapsed_seconds": time.time() - start,
            "windows": "6200bp promoter -5000/+1200; terminator -1200/+5000; transcript strand; longest mRNA; all CDS masked",
            "score": "ALT-minus-REF log10(TPM+0.1), unrounded; magnitude for prioritization",
            "training_overlap": "Bnapus cross-species transfer; homolog overlap unresolved; no family-heldout claim",
        },
        indent=2,
    )
    + "\n"
)
print("PASS nemo production")

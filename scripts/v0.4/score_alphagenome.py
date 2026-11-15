"""Restartable paired public-sequence AlphaGenome API predictions, credentials private."""

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
from alphagenome.models.v1 import dna_client

from popgenlm.regulatory import ALIASES, context
from popgenlm.regulatory_transfer import (
    allele_sequences,
    cache_key,
    central_values,
    transfer_magnitude,
)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--gene-index", type=int)
    a = p.parse_args()
    run = Path(a.run)
    out = run / "results/alphagenome"
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "cache"
    cache.mkdir(exist_ok=True)
    protocol = json.loads((run / "inputs/model_protocol.json").read_text())
    freeze = json.loads((run / "inputs/alphagenome_freeze.json").read_text())
    assert sha(run / "inputs/model_protocol.json") == freeze["protocol_sha256"]
    assert sha(run / "inputs/alphagenome_tracks.tsv") == freeze["tracks_sha256"]
    expected = pd.read_csv(run / "inputs/alphagenome_tracks.tsv", sep="\t")
    for name, m in json.loads((run / "inputs/frozen/freeze.json").read_text())["files"].items():
        assert sha(run / "inputs/frozen" / name) == m["sha256"]
    x = pd.read_csv(
        run
        / ("inputs/smoke_candidates.tsv" if a.smoke else "inputs/frozen/alphagenome_subset.tsv"),
        sep="\t",
    )
    if a.smoke:
        x = x.head(1).assign(gene="technical_fixture")
    elif a.gene_index is not None:
        x = x[x.gene == sorted(x.gene.unique())[a.gene_index]]
    seq = {}
    name = None
    with Path(
        "/projects/ag-demeaux/tali/PopGenLM/data/reference/TAIR10.1_genomic.fna"
    ).open() as fasta:
        for line in fasta:
            if line.startswith(">"):
                name = line[1:].split()[0]
                seq[name] = []
            else:
                seq[name].append(line.strip().upper())
    seq = {k: "".join(v) for k, v in seq.items()}
    key = os.environ["ALPHA_GENOME_API_KEY"]
    model = dna_client.create(key, model_version=dna_client.ModelVersion.ALL_FOLDS, timeout=60)
    specification = {
        k: freeze[k] for k in ["api", "model_version", "sdk", "organism", "output", "tracks_sha256"]
    }
    specification["sequence_length"] = protocol["alpha_sequence_bp"]
    requests = []

    def predict(s):
        identity = cache_key(s, specification)
        path = cache / (identity + ".npz")
        manifest = cache / (identity + ".json")
        if path.exists():
            m = json.loads(manifest.read_text())
            assert sha(path) == m["sha256"] and m["specification"] == specification
            print("CACHE", identity[:12], flush=True)
            with np.load(path) as saved:
                return saved["central"], int(saved["resolution"])
        for attempt in range(4):
            try:
                started = time.time()
                prediction = model.predict_sequence(
                    sequence=s,
                    organism=dna_client.Organism.HOMO_SAPIENS,
                    requested_outputs=[dna_client.OutputType.RNA_SEQ],
                    ontology_terms=None,
                )
                track = prediction.rna_seq
                assert len(track.metadata) == len(expected)
                assert track.metadata["name"].tolist() == expected["name"].tolist(), (
                    "track order/name mismatch"
                )
                central = central_values(
                    track.values, track.resolution, length=protocol["alpha_sequence_bp"]
                )
                assert central.shape[1] == len(expected)
                temp = cache / (identity + ".tmp.npz")
                np.savez_compressed(temp, central=central, resolution=track.resolution)
                temp.replace(path)
                m = {
                    "sha256": sha(path),
                    "sequence_sha256": hashlib.sha256(s.encode()).hexdigest(),
                    "specification": specification,
                    "shape": list(central.shape),
                    "resolution": track.resolution,
                    "elapsed_seconds": time.time() - started,
                    "job_id": os.environ.get("SLURM_JOB_ID"),
                    "central_only": True,
                    "attempt": attempt,
                }
                manifest.write_text(json.dumps(m, indent=2))
                requests.append(m)
                print("REQUEST_OK", identity[:12], m["elapsed_seconds"], central.shape, flush=True)
                return central, int(track.resolution)
            except Exception as e:  # noqa: BLE001 — sanitize every API/transport error before logging
                message = str(e).replace(key, "[REDACTED]")
                code = e.code().name if hasattr(e, "code") else type(e).__name__
                error = {
                    "type": type(e).__name__,
                    "code": code,
                    "error": message,
                    "attempt": attempt,
                    "job_id": os.environ.get("SLURM_JOB_ID"),
                    "sequence_sha256": hashlib.sha256(s.encode()).hexdigest(),
                }
                with (out / f"errors_{os.environ.get('SLURM_JOB_ID', 'local')}.jsonl").open(
                    "a"
                ) as f:
                    f.write(json.dumps(error) + "\n")
                print("API_ERROR", code, message, flush=True)
                if (
                    code
                    not in ["UNAVAILABLE", "DEADLINE_EXCEEDED", "RESOURCE_EXHAUSTED", "INTERNAL"]
                    or attempt == 3
                ):
                    raise RuntimeError(
                        "AlphaGenome request stopped; sanitized exact error saved"
                    ) from None
                time.sleep([10, 30, 60][attempt])

    for gene, g in x.groupby("gene", sort=True):
        dest = out / (f"{gene}.tsv" if not a.smoke else "smoke.tsv")
        done = dest.with_suffix(".complete.json")
        if done.exists():
            assert sha(dest) == json.loads(done.read_text())["sha256"]
            continue
        rows = []
        for r in g.itertuples():
            s, i = context(seq[ALIASES[str(r.chrom)]], r.pos, r.ref, protocol["alpha_sequence_bp"])
            ref, alt = allele_sequences(s, i, r.ref, r.alt)
            ref_values, res = predict(ref)
            if a.smoke:
                # One authenticated public-sequence request; no pilot predictions here.
                assert np.isfinite(ref_values).all()
                rows.append(
                    {
                        "variant_id": r.variant_id,
                        "reference_sha256": hashlib.sha256(ref.encode()).hexdigest(),
                        "resolution": res,
                        "tracks": ref_values.shape[1],
                        "smoke": True,
                    }
                )
                break
            alt_values, alt_res = predict(alt)
            assert res == alt_res
            rows.append(
                {
                    "gene": gene,
                    "variant_id": r.variant_id,
                    "score": transfer_magnitude(ref_values, alt_values),
                    "ref_cache": cache_key(ref, specification),
                    "alt_cache": cache_key(alt, specification),
                    "resolution": res,
                    "tracks": ref_values.shape[1],
                    "context_bp": len(s),
                }
            )
            print("SCORED", gene, r.variant_id, len(rows), len(g), flush=True)
        pd.DataFrame(rows).to_csv(dest, sep="\t", index=False)
        done.write_text(
            json.dumps(
                {
                    "sha256": sha(dest),
                    "rows": len(rows),
                    "job_id": os.environ.get("SLURM_JOB_ID"),
                    "new_requests": len(requests),
                    "specification": specification,
                },
                indent=2,
            )
        )
    (out / ("smoke_manifest.json" if a.smoke else f"manifest_{a.gene_index}.json")).write_text(
        json.dumps(
            {
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "python": sys.executable,
                "sdk": metadata.version("alphagenome"),
                "specification": specification,
                "requests": requests,
                "orientation": "paired genomic REF and ALT; magnitude only; mammalian cross-species transfer",
            },
            indent=2,
        )
    )
    print("PASS AlphaGenome", a.smoke, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 — prevent credential-bearing exception traces
        key = os.environ.get("ALPHA_GENOME_API_KEY", "")
        msg = str(exc)
        msg = msg.replace(key, "[REDACTED]") if key else msg
        print("STOP", type(exc).__name__, msg, flush=True)
        raise SystemExit(1)

#!/usr/bin/env python3
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT / "src"))
from popgenlm.plantcad import reverse_complement

ATOL = 1e-4
RTOL = 1e-4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream-zero-shot-score", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    CK, FIX, REF, OUT, UPSTREAM = (
        args.checkpoint,
        args.fixture,
        args.reference,
        args.output,
        args.upstream_zero_shot_score,
    )
    print("script_dir", SCRIPT_DIR)
    for required in (CK, FIX, REF, UPSTREAM):
        assert required.exists(), required
    assert OUT.parent.exists(), OUT.parent
    assert len(list(csv.DictReader(FIX.open(), delimiter="\t"))) == 100
    if args.preflight_only:
        print("PREFLIGHT_PASS")
        return
    import torch
    from plantcad_reference_compare import score as author_score

    from popgenlm.plantcad import load_local_model

    assert (
        torch.cuda.is_available()
        and torch.__version__ == "2.5.1+cu124"
        and torch.version.cuda == "12.4"
        and torch._C._GLIBCXX_USE_CXX11_ABI is False
    )
    seqs = {}
    name = None
    chunks = []
    for line in REF.open():
        if line.startswith(">"):
            if name:
                seqs[name] = "".join(chunks).upper()
            name = line[1:].split()[0]
            chunks = []
        else:
            chunks.append(line.strip())
    if name:
        seqs[name] = "".join(chunks).upper()
    variants = list(csv.DictReader(FIX.open(), delimiter="\t"))
    assert len(variants) == 100 and len({v["variant_id"] for v in variants}) == 100
    contexts = []
    for v in variants:
        pos = int(v["pos"])
        seq = seqs[v["fasta_chrom"]]
        c = seq[pos - 1 - 255 : pos - 1 - 255 + 512]
        assert len(c) == 512 and c[255] == v["ref"]
        contexts.append(c)
    t = time.perf_counter()
    model, tok = load_local_model(str(CK))
    torch.cuda.synchronize()
    print("model_load_seconds", time.perf_counter() - t)
    vocab = tok.get_vocab()
    ids = {b: vocab[b.lower()] for b in "ACGT"}
    mask = tok.mask_token_id

    def adapter(ctxs, target, batch):
        allp = []
        for start in range(0, len(ctxs), batch):
            chunk = ctxs[start : start + batch]
            enc = tok(chunk, add_special_tokens=False, return_tensors="pt")
            x = enc["input_ids"]
            assert x.shape == (len(chunk), 512)
            for i, c in enumerate(chunk):
                assert int(x[i, target]) == ids[c[target]]
                x[i, target] = mask
                assert (x[i] == mask).sum().item() == 1
            with torch.inference_mode():
                z = model(x.to("cuda")).logits[:, target, :][:, [ids[b] for b in "ACGT"]].float()
                torch.cuda.synchronize()
            allp.extend(torch.log_softmax(z, 1).cpu().tolist())
        return allp

    p1 = adapter(contexts, 255, 1)
    t = time.perf_counter()
    p8 = adapter(contexts, 255, 8)
    torch.cuda.synchronize()
    print("batch8_seconds", time.perf_counter() - t)
    ref = author_score(
        model,
        tok,
        contexts,
        [{"ref": v["ref"], "alt": v["alt"]} for v in variants],
        "cuda",
        1,
        255,
        args.upstream_zero_shot_score,
    )
    rcctx = [reverse_complement(c) for c in contexts]
    rcvars = [
        {
            "ref": {"A": "T", "C": "G", "G": "C", "T": "A"}[v["ref"]],
            "alt": {"A": "T", "C": "G", "G": "C", "T": "A"}[v["alt"]],
        }
        for v in variants
    ]
    prc = adapter(rcctx, 256, 1)
    rows = []
    fails = []
    for i, v in enumerate(variants):
        raw = lambda p, ref=v["ref"], alt=v["alt"]: p["ACGT".index(alt)] - p["ACGT".index(ref)]
        a, b, r = raw(p1[i]), raw(p8[i]), float(ref[i])
        rr = prc[i]["ACGT".index(rcvars[i]["alt"])] - prc[i]["ACGT".index(rcvars[i]["ref"])]
        vals = [a, b, r, rr]
        assert all(math.isfinite(x) for x in vals)
        checks = [
            abs(a - r) <= ATOL + RTOL * abs(r),
            abs(b - a) <= ATOL + RTOL * abs(a),
            abs(rr - a) <= ATOL + RTOL * abs(a),
        ]
        if not all(checks):
            fails.append(v["variant_id"])
        rows.append(
            {
                "variant_id": v["variant_id"],
                "adapter_batch1": a,
                "authors_reference": r,
                "adapter_batch8": b,
                "reverse_complement": rr,
                "abs_ref_diff": abs(a - r),
                "abs_batch_diff": abs(b - a),
                "abs_rc_diff": abs(rr - a),
                "ref_pass": checks[0],
                "batch_pass": checks[1],
                "rc_pass": checks[2],
            }
        )
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0], delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    summary = {
        "rows": 100,
        "atol": ATOL,
        "rtol": RTOL,
        "failed_variant_ids": fails,
        "overall_pass": not fails,
        "max_abs_ref": max(r["abs_ref_diff"] for r in rows),
        "max_abs_batch": max(r["abs_batch_diff"] for r in rows),
        "max_abs_rc": max(r["abs_rc_diff"] for r in rows),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "output": str(OUT),
    }
    (OUT.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    assert not fails
    print("PASS")


if __name__ == "__main__":
    main()

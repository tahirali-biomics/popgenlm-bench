"""Restartable chromosome scoring and strict input/output validation."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

from .plantcad import coordinate_to_context, population_scores, score_contexts, validate_reference

REVISION = "e624c13c3d35415348b854c87a218893b23564f7"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(path):
    p = Path(path).resolve()
    st = p.stat()
    return {"path": str(p), "size": st.st_size, "mtime_ns": st.st_mtime_ns,
            "ctime_ns": st.st_ctime_ns}


def write_json(path, data):
    with Path(path).open("x") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")


def write_tsv(path, rows):
    with Path(path).open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def read_tsv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def row_key(row):
    return (str(row["chrom"]), int(row["pos"]), row["ref"], row["alt"])


def load_variants(path, mapping, expected_count):
    rows = read_tsv(path)
    if len(rows) != expected_count or not rows:
        raise ValueError("unexpected variant count")
    seen, keys = set(), set()
    for row in rows:
        chrom, pos, ref, alt = row_key(row)
        if chrom not in mapping or row["fasta_chrom"] != mapping[chrom]["fasta_chrom"]:
            raise ValueError("chromosome mapping mismatch")
        if not 1 <= pos <= int(mapping[chrom]["length"]):
            raise ValueError("coordinate outside chromosome")
        population_scores(ref, alt, row["af_alt"], 0.0)
        if not math.isclose(float(row["af_alt"]), int(row["ac_alt"]) / int(row["an"]),
                            abs_tol=1e-12, rel_tol=1e-12):
            raise ValueError("allele frequency/count mismatch")
        identifier = row.get("variant_id") or f"{chrom}:{pos}:{ref}:{alt}"
        if identifier in seen or row_key(row) in keys:
            raise ValueError("duplicate variant ID or key")
        seen.add(identifier)
        keys.add(row_key(row))
        row["variant_id"] = identifier
    return sorted(rows, key=lambda r: (int(r["chrom"]), int(r["pos"]), r["ref"],
                                       r["alt"], r["variant_id"]))


def validate_scores(rows, expected):
    if len(rows) != len(expected):
        raise ValueError("missing score rows")
    by_id = {}
    for row in rows:
        identifier = row["variant_id"]
        if identifier in by_id:
            raise ValueError("duplicate score ID")
        by_id[identifier] = row
    if set(by_id) != {r["variant_id"] for r in expected}:
        raise ValueError("missing/unexpected score IDs")
    for original in expected:
        row = by_id[original["variant_id"]]
        for key, value in original.items():
            if str(row[key]) != str(value):
                raise ValueError(f"changed original metadata: {key}")
        raw = float(row["plantcad_score_ref_alt"])
        want = population_scores(original["ref"], original["alt"], original["af_alt"], raw)
        for field, val in want.items():
            got = row[field]
            if val is None:
                if got not in (None, ""):
                    raise ValueError("frequency tie must have undefined minor/major")
            elif isinstance(val, (int, float)) and not isinstance(val, bool):
                if not math.isfinite(float(got)) or float(got) != val:
                    raise ValueError(f"incorrect/nonfinite score orientation: {field}")
            elif str(got) != str(val):
                raise ValueError(f"incorrect allele orientation: {field}")
        for field in ("ref_logprob", "alt_logprob"):
            if not math.isfinite(float(row[field])):
                raise ValueError("nonfinite log probability")
        if abs(raw - (float(row["alt_logprob"]) - float(row["ref_logprob"]))) > 1e-12:
            raise ValueError("raw log-probability identity mismatch")
    return by_id


def check_resume(table, marker, fingerprint, expected):
    table, marker = Path(table), Path(marker)
    if not table.exists() and not marker.exists():
        return None
    if not table.is_file() or not marker.is_file():
        raise ValueError("partial chromosome output preserved; manual inspection required")
    meta = json.loads(marker.read_text())
    if meta["fingerprint"] != fingerprint or meta["sha256"] != digest(table):
        raise ValueError("completed output inputs/configuration/hash mismatch")
    rows = read_tsv(table)
    validate_scores(rows, expected)
    if meta["rows"] != len(rows):
        raise ValueError("completion row count mismatch")
    return rows


def setup_offline(config):
    cache = Path(config["cache_dir"]).resolve()
    if Path(config["project_root"]).resolve() not in cache.parents:
        raise ValueError("cache must be inside project")
    for variable, subdir in (("HF_HOME", "home"), ("HF_HUB_CACHE", "hub"),
                            ("HUGGINGFACE_HUB_CACHE", "hub"), ("HF_MODULES_CACHE", "modules"),
                            ("HF_ASSETS_CACHE", "assets"), ("XDG_CACHE_HOME", "xdg"),
                            ("MPLCONFIGDIR", "matplotlib")):
        os.environ[variable] = str(cache / subdir)
    os.environ.pop("TRANSFORMERS_CACHE", None)
    for variable in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[variable] = "1"


def preflight(config_path, import_runtime=True):
    config_path = Path(config_path).resolve()
    c = json.loads(config_path.read_text())
    if c["checkpoint_revision"] != REVISION:
        raise ValueError("wrong pinned revision")
    if (c["dtype"], c["batch_size"], c["context_length"], c["target_index"]) != (
            "float32", 8, 512, 255):
        raise ValueError("unexpected scoring configuration")
    if (c["atol"], c["rtol"]) != (1e-4, 1e-4):
        raise ValueError("changed comparison tolerances")
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        raise ValueError("use Python -I -B")
    if Path(sys.executable) != Path(c["python"]):
        raise ValueError("wrong interpreter")
    for path, want in c["hashes"].items():
        if digest(path) != want:
            raise ValueError(f"file/code hash changed: {path}")
    for record in c["large_file_identities"]:
        if identity(record["path"]) != record:
            raise ValueError("previously verified large input changed")
    mapping = {r["chrom"]: r for r in read_tsv(c["mapping"])}
    rows = load_variants(c["input"], mapping, c["expected_rows"])
    out = Path(c["output_dir"]).resolve()
    repo = Path(c["repo"]).resolve()
    if not out.is_dir() or not os.access(out, os.W_OK) or repo == out or repo in out.parents:
        raise ValueError("output must be an existing writable directory outside repo")
    setup_offline(c)
    locations = {"production": __file__}
    if import_runtime:
        import torch
        import torchvision
        import torchaudio
        import transformers
        import causal_conv1d
        import mamba_ssm
        if torch.__version__ != "2.5.1+cu124" or torch.version.cuda != "12.4":
            raise ValueError("PyTorch/CUDA build changed")
        if torch._C._GLIBCXX_USE_CXX11_ABI is not False:
            raise ValueError("CXX11 ABI changed")
        for mod in (torch, torchvision, torchaudio, transformers, causal_conv1d, mamba_ssm):
            locations[mod.__name__] = str(Path(mod.__file__).resolve())
            if Path(sys.prefix) not in Path(mod.__file__).resolve().parents:
                raise ValueError("dependency imported outside isolated environment")
    fingerprint = digest(config_path)
    for chrom in sorted(mapping, key=int):
        subset = [r for r in rows if r["chrom"] == chrom]
        if subset:
            check_resume(out / f"chr{chrom}.tsv", out / f"chr{chrom}.complete.json",
                         fingerprint, subset)
    report = {"status": "PREFLIGHT_PASS", "rows": len(rows),
              "chromosomes": {k: sum(r["chrom"] == k for r in rows) for k in mapping},
              "frequency_ties": sum(float(r["af_alt"]) == 0.5 for r in rows),
              "python": sys.executable, "module_locations": locations,
              "config_sha256": fingerprint, "output_dir": str(out),
              "weights_loaded": False, "gpu_work_executed": False}
    return c, rows, mapping, report


def load_reference(path, mapping):
    wanted = {m["fasta_chrom"]: int(m["length"]) for m in mapping.values()}
    seqs, name = {}, None
    with Path(path).open() as f:
        for line in f:
            if line.startswith(">"):
                name = line[1:].split()[0]
                if name in wanted:
                    seqs[name] = []
            elif name in wanted:
                seqs[name].append(line.strip().upper())
    seqs = {k: "".join(v) for k, v in seqs.items()}
    if {k: len(v) for k, v in seqs.items()} != wanted:
        raise ValueError("reference chromosome length mismatch")
    return seqs


def compare_overlap(all_rows, comparison, atol, rtol):
    by_id = {r["variant_id"]: r for r in all_rows}
    if len(by_id) != len(all_rows):
        raise ValueError("duplicate production IDs")
    archived = read_tsv(comparison)
    if len(archived) != 100 or len({r["variant_id"] for r in archived}) != 100:
        raise ValueError("invalid archived comparison")
    comparisons = []
    for row in archived:
        identifier = row["variant_id"]
        if identifier not in by_id:
            raise ValueError("missing overlap variant")
        observed = float(by_id[identifier]["plantcad_score_ref_alt"])
        result = {"variant_id": identifier, "production_batch8": observed}
        for field in ("adapter_batch1", "adapter_batch8", "authors_reference"):
            expected = float(row[field])
            if not (math.isfinite(observed) and math.isfinite(expected)):
                raise ValueError("nonfinite overlap")
            diff = abs(observed - expected)
            result[field] = expected
            result[field + "_abs_diff"] = diff
            result[field + "_pass"] = diff <= atol + rtol * abs(expected)
        comparisons.append(result)
    summary = {field: {"rows": 100, "max_abs_difference": max(
        r[field + "_abs_diff"] for r in comparisons),
        "failed_variant_ids": [r["variant_id"] for r in comparisons if not r[field + "_pass"]]}
        for field in ("adapter_batch1", "adapter_batch8", "authors_reference")}
    return comparisons, summary


def run(config_path):
    c, variants, mapping, report = preflight(config_path)
    print(json.dumps(report), flush=True)
    import torch
    from .plantcad import load_local_model
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise ValueError("inference requires a CUDA SLURM allocation")
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(c["output_dir"])
    fp = report["config_sha256"]
    seqs = load_reference(c["reference"], mapping)
    contexts = []
    for row in variants:
        sequence = seqs[row["fasta_chrom"]]
        validate_reference(sequence, int(row["pos"]), row["ref"])
        ctx, _ = coordinate_to_context(sequence, int(row["pos"]))
        if ctx[255] != row["ref"]:
            raise ValueError("REF context mismatch")
        contexts.append(ctx)
    software = {name: importlib.metadata.version(name) for name in
                ("torch", "torchvision", "torchaudio", "transformers", "mamba-ssm",
                 "causal-conv1d", "numpy", "pandas")}
    print("GPU", torch.cuda.get_device_name(0), "software", json.dumps(software), flush=True)
    t = time.perf_counter()
    model, tokenizer = load_local_model(c["checkpoint"])
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - t
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("non-float32 model parameter")
    list(score_contexts(model, tokenizer, contexts[:8], variants[:8], batch_size=8))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    all_rows, timings = [], {}
    for chrom in sorted(mapping, key=int):
        indexes = [i for i, v in enumerate(variants) if v["chrom"] == chrom]
        if not indexes:
            continue
        subset = [variants[i] for i in indexes]
        table, marker = out / f"chr{chrom}.tsv", out / f"chr{chrom}.complete.json"
        rows = check_resume(table, marker, fp, subset)
        if rows is None:
            start = time.perf_counter()
            rows = [{**variant, **score} for variant, score in zip(
                subset, score_contexts(model, tokenizer, [contexts[i] for i in indexes],
                                       subset, batch_size=8))]
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            validate_scores(rows, subset)
            write_tsv(table, rows)
            write_json(marker, {"fingerprint": fp, "rows": len(rows), "sha256": digest(table),
                                "scoring_seconds": seconds})
            timings[chrom] = {"rows": len(rows), "seconds": seconds, "reused": False}
        else:
            timings[chrom] = {"rows": len(rows), "seconds": 0, "reused": True}
        all_rows.extend(rows)
        print("CHROMOSOME_PASS", chrom, json.dumps(timings[chrom]), flush=True)
    validate_scores(all_rows, variants)
    combined = out / "plantcad_10000.tsv"
    if combined.exists():
        validate_scores(read_tsv(combined), variants)
        if read_tsv(combined) != [{k: "" if v is None else str(v) for k, v in r.items()}
                                  for r in all_rows]:
            raise ValueError("combined output differs from chromosome outputs")
    else:
        write_tsv(combined, all_rows)
    overlap, overlap_summary = compare_overlap(all_rows, c["comparison"], c["atol"], c["rtol"])
    write_tsv(out / "overlap_100.tsv", overlap)
    total_seconds = sum(t["seconds"] for t in timings.values())
    scored = sum(t["rows"] for t in timings.values() if not t["reused"])
    passed = all(not item["failed_variant_ids"] for item in overlap_summary.values())
    summary = {
        "status": "PASS" if passed else "FAIL", "job_id": os.environ["SLURM_JOB_ID"],
        "node": os.environ.get("SLURMD_NODENAME"), "gpu": torch.cuda.get_device_name(0),
        "python": sys.version, "executable": sys.executable, "prefix": sys.prefix,
        "software": software, "cuda_build": torch.version.cuda,
        "cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI, "dtype": "float32",
        "config_sha256": fp, "rows": len(all_rows), "unique_ids": len({r["variant_id"] for r in all_rows}),
        "model_load_seconds": load_seconds, "warmed_scoring_seconds": total_seconds,
        "warmed_variants_per_second": scored / total_seconds if total_seconds else None,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(), "chromosomes": timings,
        "frequency_ties": report["frequency_ties"], "atol": c["atol"], "rtol": c["rtol"],
        "overlap": overlap_summary, "score_sha256": digest(combined),
        "all_ref_and_token_checks_passed": True, "all_scores_finite_and_oriented": True,
    }
    write_json(out / "run_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if not passed:
        raise ValueError("100-variant overlap tolerance failure; diagnostics preserved")
    print("PASS", flush=True)

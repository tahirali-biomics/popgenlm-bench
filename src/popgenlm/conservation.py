"""Streaming lookup of per-base conservation scores in bedGraph files."""

from __future__ import annotations

import csv
import gzip
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, TextIO


# Explicit mappings are intentional: RefSeq accessions are not derivable from
# the numeric VCF chromosome labels without an assembly-specific contract.
CHROM_MAP = {
    "1": "NC_003070.9", "2": "NC_003071.7", "3": "NC_003074.8",
    "4": "NC_003075.7", "5": "NC_003076.8",
}
TRACK_CHROM_MAP = {f"Chr{i}": fasta for i, fasta in CHROM_MAP.items()}
EXCLUDED_TRACK_CHROMS = frozenset({"ChrC", "ChrM"})
CHROM_LENGTHS = {
    "NC_003070.9": 30427671, "NC_003071.7": 19698289,
    "NC_003074.8": 23459830, "NC_003075.7": 18585056,
    "NC_003076.8": 26975502,
}


class TrackContractError(ValueError):
    """The track violates the declared bedGraph/assembly contract."""


def _parse_track_row(line: str, line_number: int):
    fields = line.rstrip("\n").split()
    if len(fields) != 4:
        raise TrackContractError(f"line {line_number}: expected 4 bedGraph fields")
    chrom, start_s, end_s, score_s = fields
    if chrom in TRACK_CHROM_MAP:
        fasta_chrom = TRACK_CHROM_MAP[chrom]
    elif chrom in EXCLUDED_TRACK_CHROMS:
        fasta_chrom = None
    else:
        raise TrackContractError(f"line {line_number}: unknown chromosome {chrom}")
    try:
        start, end, score = int(start_s), int(end_s), float(score_s)
    except ValueError as exc:
        raise TrackContractError(f"line {line_number}: invalid coordinate/score") from exc
    if start < 0 or end <= start:
        raise TrackContractError(f"line {line_number}: invalid interval {start}:{end}")
    if fasta_chrom is not None and end > CHROM_LENGTHS[fasta_chrom]:
        raise TrackContractError(f"line {line_number}: interval exceeds {chrom} bound")
    if not math.isfinite(score):
        raise TrackContractError(f"line {line_number}: non-finite score")
    return chrom, fasta_chrom, start, end, score


def _read_variants(path: str | Path):
    rows = []
    seen = set()
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"chrom", "pos", "ref", "alt"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError("variant table must contain chrom, pos, ref, alt")
        for line_number, row in enumerate(reader, 2):
            chrom = row["chrom"]
            if chrom not in CHROM_MAP:
                raise ValueError(f"line {line_number}: unknown variant chromosome {chrom}")
            pos = int(row["pos"])
            key = (CHROM_MAP[chrom], pos, row["ref"], row["alt"])
            if key in seen:
                raise ValueError(f"duplicate normalized variant key: {key}")
            if not 1 <= pos <= CHROM_LENGTHS[key[0]]:
                raise ValueError(f"variant outside chromosome bounds: {key}")
            seen.add(key)
            rows.append({**row, "fasta_chrom": key[0], "_pos": pos, "_key": key})
    rows.sort(key=lambda row: (row["fasta_chrom"], row["_pos"], row["ref"], row["alt"]))
    return rows


def lookup_bedgraph(
    variant_path: str | Path,
    track_path: str | Path,
    output_path: str | Path,
    metadata_path: str | Path | None = None,
) -> dict:
    """Look up variants while streaming the compressed bedGraph exactly once.

    bedGraph intervals are zero-based half-open; variant positions are one-based.
    The score at variant POS is selected when start <= POS-1 < end. Missing
    coverage is emitted as an empty score and status ``missing_coverage``.
    """
    variants = _read_variants(variant_path)
    by_chrom = {}
    for row in variants:
        by_chrom.setdefault(row["fasta_chrom"], []).append(row)
    results = {row["_key"]: {**row, "phylop": None, "status": "missing_coverage"} for row in variants}
    previous = {}
    pointers = {chrom: 0 for chrom in by_chrom}
    interval_counts = Counter()
    covered_bases = Counter()
    excluded_interval_counts = Counter()
    with gzip.open(track_path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip() or line.startswith("track") or line.startswith("#"):
                continue
            track_chrom, fasta_chrom, start, end, score = _parse_track_row(line, line_number)
            old = previous.get(track_chrom)
            if old is not None and start < old[1]:
                raise TrackContractError(f"line {line_number}: interval order/overlap on {track_chrom}")
            previous[track_chrom] = (start, end)
            if fasta_chrom is None:
                excluded_interval_counts[track_chrom] += 1
                continue
            interval_counts[fasta_chrom] += 1
            covered_bases[fasta_chrom] += end - start
            rows = by_chrom.get(fasta_chrom, ())
            pointer = pointers.get(fasta_chrom, 0)
            while pointer < len(rows) and rows[pointer]["_pos"] - 1 < start:
                pointer += 1
            while pointer < len(rows) and rows[pointer]["_pos"] - 1 < end:
                row = rows[pointer]
                result = results[row["_key"]]
                result["phylop"] = score
                result["status"] = "covered"
                pointer += 1
            pointers[fasta_chrom] = pointer
    fields = ["fasta_chrom", "pos", "ref", "alt", "phylop", "status"]
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in variants:
            out = results[row["_key"]]
            writer.writerow({field: out[field] if field != "phylop" or out[field] is not None else "" for field in fields})
    summary = {
        "variant_rows": len(variants),
        "unique_normalized_keys": len(variants),
        "covered_rows": sum(r["status"] == "covered" for r in results.values()),
        "missing_coverage_rows": sum(r["status"] == "missing_coverage" for r in results.values()),
        "track_interval_counts": dict(interval_counts),
        "excluded_non_nuclear_interval_counts": dict(excluded_interval_counts),
        "track_contract_scope": "Chr1..Chr5 bound-checked; ChrC/ChrM syntax and order checked but excluded from lookup",
        "track_covered_bases_sum": dict(covered_bases),
        "track_passes_contract": True,
        "coordinate_convention": "bedGraph zero-based half-open; variants one-based",
        "chromosome_map": CHROM_MAP,
    }
    if metadata_path is not None:
        import json
        Path(metadata_path).write_text(json.dumps(summary, indent=2) + "\n")
    return summary

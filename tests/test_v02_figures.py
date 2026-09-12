"""Release-invariant tests for the PopGenLM Bench v0.2 figures."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "reports" / "v0.2" / "figures"

MAF = FIGURE_DIR / "maf-spectrum.png"
MAF_SCORE = FIGURE_DIR / "score-vs-maf.png"
CLASS = FIGURE_DIR / "score-by-frequency-class.png"
ORIENTATION = FIGURE_DIR / "orientation-check.png"
OVERVIEW_PNG = FIGURE_DIR / "v02_population_gpn_overview.png"
OVERVIEW_PDF = FIGURE_DIR / "v02_population_gpn_overview.pdf"
METADATA = FIGURE_DIR / "figure_metadata.json"

EXPECTED_ARTIFACT_HASHES = {
    "maf-spectrum.png": ("9f4a0c1648a2ff167e4590bbdc1344f23dac39962ceb84ec52443c843e248748"),
    "score-vs-maf.png": ("a2e3a441b3cd4fa4c6fda3c46e995577c412073bcd7bc739f46811c7478c3fcc"),
    "score-by-frequency-class.png": (
        "e05517898e5e52423fb7c8e709f903d50d5c38e5af8728ab0bbca7d755470336"
    ),
    "orientation-check.png": ("3694b1334857e763a3e3d88d3d930b748a7777365ce1544117f7130a84479270"),
    "v02_population_gpn_overview.png": (
        "d33e1695d26a7809c1f039150324099225bff07c65db08a4d0e41181ba040237"
    ),
    "v02_population_gpn_overview.pdf": (
        "1953adc7e335f4c5704b0baf4ecd3fd9e1a161ae194b018a7320aff1d45e7185"
    ),
    "figure_metadata.json": ("b66e5b5e307e3e602483643325be4ab63dbae4fe81828f12fc9050c1748e9d55"),
}

EXPECTED_OUTPUT_HASHES = {
    filename: digest
    for filename, digest in EXPECTED_ARTIFACT_HASHES.items()
    if filename != "figure_metadata.json"
}

EXPECTED_INPUT_HASHES = {
    "joined_data": ("ef188e83d0e598d016c158b4daeec08be63e19723840fc42b494c76443fca469"),
    "analysis": ("c41e2a4e59c5a9fc25f12803ae30149c502096167328874b0890e1bf4b5f11e9"),
    "class_summary": ("a7dfd82ee5f0850a68dbfcdfc21b47a0df0a694d22e4ecfc462d75afc3785c3e"),
    "pairwise": ("3779fe5578ca4b33b5d66c4a34d4eefe61d85417982a4e5f45e31b3571f80739"),
    "sensitivity": ("d00b1a016f87102d45cbf276b08b0fa0fea3379ab08194c14a3f4e8091e2a3fa"),
    "chromosome": ("5f0791cf974a35524393d99cdc248dede8fac0bc25978aca7dbefe1dd2e74dc3"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        signature = handle.read(8)

        if signature != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"Not a PNG file: {path}")

        length = struct.unpack(">I", handle.read(4))[0]
        chunk_type = handle.read(4)

        if chunk_type != b"IHDR" or length < 8:
            raise ValueError(f"Invalid PNG IHDR: {path}")

        width, height = struct.unpack(
            ">II",
            handle.read(8),
        )

    return width, height


def test_v02_figure_artifact_hashes() -> None:
    paths = {
        MAF.name: MAF,
        MAF_SCORE.name: MAF_SCORE,
        CLASS.name: CLASS,
        ORIENTATION.name: ORIENTATION,
        OVERVIEW_PNG.name: OVERVIEW_PNG,
        OVERVIEW_PDF.name: OVERVIEW_PDF,
        METADATA.name: METADATA,
    }

    for filename, path in paths.items():
        assert path.exists()
        assert sha256_file(path) == EXPECTED_ARTIFACT_HASHES[filename]


def test_v02_figure_metadata() -> None:
    metadata = json.loads(METADATA.read_text())

    assert metadata["figure_set"] == ("popgenlm_v0.2_population_gpn")

    assert metadata["input_hashes"] == EXPECTED_INPUT_HASHES

    assert metadata["outputs"] == EXPECTED_OUTPUT_HASHES

    assert set(metadata["panels"]) == {
        "A",
        "B",
        "C",
        "D",
    }

    environment = metadata["environment"]

    assert environment["matplotlib"] == "3.11.2"
    assert environment["numpy"] == "2.5.3"
    assert environment["pandas"] == "3.0.5"


def test_v02_png_artifacts_are_valid() -> None:
    pngs = [
        MAF,
        MAF_SCORE,
        CLASS,
        ORIENTATION,
        OVERVIEW_PNG,
    ]

    dimensions = {path.name: png_dimensions(path) for path in pngs}

    for width, height in dimensions.values():
        assert width > 1000
        assert height > 1000

    overview_width, overview_height = dimensions[OVERVIEW_PNG.name]

    assert overview_width > overview_height


def test_v02_pdf_artifact_is_valid() -> None:
    with OVERVIEW_PDF.open("rb") as handle:
        header = handle.read(8)

    assert header.startswith(b"%PDF-")
    assert OVERVIEW_PDF.stat().st_size > 100_000

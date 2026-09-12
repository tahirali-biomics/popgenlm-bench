"""Release-invariant tests for the PopGenLM Bench v0.2 Colab notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

NOTEBOOK = ROOT / "notebooks" / "popgenlm-bench-v0.2.ipynb"

EXPECTED_SHA256 = "d328fd69c6bb02cd8753cbd15404fdc0fed6591ffe049d5ca15f3f5945b89248"

PINNED_FALLBACK_COMMIT = "012975df95c69648f01eb7c19ee94f8d04e7782e"

EXPECTED_ARTIFACT_HASHES = {
    "benchmark": ("ef188e83d0e598d016c158b4daeec08be63e19723840fc42b494c76443fca469"),
    "analysis": ("c41e2a4e59c5a9fc25f12803ae30149c502096167328874b0890e1bf4b5f11e9"),
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


def load_notebook() -> dict:
    return json.loads(NOTEBOOK.read_text())


def all_source(notebook: dict) -> str:
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"])


def test_v02_colab_hash_is_frozen() -> None:
    assert NOTEBOOK.exists()
    assert sha256_file(NOTEBOOK) == EXPECTED_SHA256


def test_v02_colab_structure() -> None:
    notebook = load_notebook()

    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] == 5

    cells = notebook["cells"]

    assert len(cells) == 18

    markdown_cells = [cell for cell in cells if cell["cell_type"] == "markdown"]

    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]

    assert len(markdown_cells) == 9
    assert len(code_cells) == 9

    for cell in code_cells:
        assert cell["execution_count"] is None
        assert cell["outputs"] == []


def test_v02_colab_code_cells_compile() -> None:
    notebook = load_notebook()

    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

    for index, cell in enumerate(
        code_cells,
        start=1,
    ):
        compile(
            "".join(cell["source"]),
            f"v02-colab-cell-{index}",
            "exec",
        )


def test_v02_colab_uses_release_and_pinned_fallback() -> None:
    notebook = load_notebook()
    source = all_source(notebook)

    assert 'RELEASE_REF = "v0.2.0"' in source

    assert f'DEVELOPMENT_REF = "{PINNED_FALLBACK_COMMIT}"' in source

    # The released notebook must not fall back to a mutable
    # development branch.
    assert "v0.2-population-benchmark" not in source


def test_v02_colab_freezes_all_downloaded_artifacts() -> None:
    notebook = load_notebook()
    source = all_source(notebook)

    for digest in EXPECTED_ARTIFACT_HASHES.values():
        assert digest in source

    assert "1001g_population_10000_gpn.tsv" in source

    assert "population_gpn_analysis.json" in source

    assert "frequency_class_pairwise.tsv" in source

    assert "sensitivity_summary.tsv" in source

    assert "chromosome_robustness.tsv" in source


def test_v02_colab_preserves_interpretation_boundary() -> None:
    notebook = load_notebook()
    source = all_source(notebook)

    assert "gpn_score_minor_vs_major" in source

    assert "log P(minor) - log P(major)" in source

    assert "not a" in source.lower() and "test of natural selection" in source.lower()

    assert "weak but reproducible concordance" in source
